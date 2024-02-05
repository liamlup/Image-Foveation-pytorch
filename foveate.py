import torch
import torch.nn.functional as F

"""
This script is a PyTorch implementation of foveated image generation.
It is an implementation based on the following papers:

Perry, Jeffrey S., and Wilson S. Geisler. "Gaze-contingent real-time simulation of arbitrary visual fields." 
Human vision and electronic imaging VII. Vol. 4662. International Society for Optics and Photonics, 2002.

Jiang, Ming, et al. "Salicon: Saliency in context." 
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

and the following repository:
https://github.com/ouyangzhibo/Image_Foveation_Python

"""


def genGaussiankernel(width, sigma):
    x = torch.arange(-int(width / 2), int(width / 2) + 1, 1, dtype=torch.float32)
    x2d, y2d = torch.meshgrid(x, x, indexing='xy')
    kernel_2d = torch.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / torch.sum(kernel_2d)
    return kernel_2d


def pyramid(img, sigma: float = 1, prNum=6):
    im_tensor = img.clone().float()
    height_ori, width_ori = im_tensor.shape[1], im_tensor.shape[2]
    G = im_tensor.clone()
    pyramids = [G]

    Gaus_kernel2D = genGaussiankernel(5, sigma)
    Gaus_kernel2D = Gaus_kernel2D.repeat(3, 1, 1, 1)  # Adjusting for 3 channels

    # Down sample
    for i in range(1, prNum):
        G = F.conv2d(G.unsqueeze(0), Gaus_kernel2D, padding=2, groups=3)
        G = F.interpolate(G, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=False).squeeze(0)
        pyramids.append(G)

    # Up sample
    for i in range(1, prNum):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i - 1:
                im_size = (curr_im.shape[1] * 2, curr_im.shape[2] * 2)
            else:
                im_size = (height_ori, width_ori)
            curr_im = F.interpolate(curr_im.unsqueeze(0), size=im_size, mode='bilinear', align_corners=False).squeeze(0)
            curr_im = F.conv2d(curr_im.unsqueeze(0), Gaus_kernel2D, padding=2, groups=3).squeeze(0)

        pyramids[i] = curr_im

    return torch.stack(pyramids, dim=0)


def foveate_image(img, fixs):
    """
    img: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]

    This function outputs the foveated image with given input image and fixations.
    """
    # TODO: Use visual angle instead of pixels for fixations

    sigma = 0.248
    prNum = 6
    As = pyramid(img, sigma, prNum)
    _, height, width = img.shape

    # compute coefficients
    p = 7.5
    k = 3
    alpha = 2.5

    x = torch.arange(0, width, dtype=torch.float32)
    y = torch.arange(0, height, dtype=torch.float32)
    x2d, y2d = torch.meshgrid(x, y, indexing='xy')
    theta = torch.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = torch.min(theta, torch.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)

    Ts = []
    for i in range(1, prNum):
        Ts.append(torch.exp(-((2 ** (i - 3)) * R / sigma) ** 2 * k))
    Ts.append(torch.zeros_like(theta))

    # Omega
    omega = torch.zeros(prNum)
    for i in range(1, prNum):
        omega[i - 1] = torch.sqrt(torch.log(torch.tensor(2)) / k) / (2 ** (i - 3)) * sigma

    omega[omega > 1] = 1

    # Layer index
    layer_ind = torch.zeros_like(R)
    for i in range(1, prNum):
        ind = torch.logical_and(torch.ge(R, omega[i]), torch.le(R, omega[i - 1]))
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i - 1] - Ts[i] + 1e-5))

    # M
    Ms = torch.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = torch.isclose(layer_ind, torch.tensor(i, dtype=torch.float32))
        if torch.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i - 1][ind]

        ind = torch.isclose(layer_ind - 1, torch.tensor(i, dtype=torch.float32))
        if torch.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    print('num of full-res pixel', torch.sum(torch.isclose(Ms[0], torch.tensor(1, dtype=torch.float32))))
    # generate periphery image
    im_fov = torch.zeros_like(As[0], dtype=torch.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            Mt = M
            At = A[i]
            im_fov[i, :, :] += torch.mul(Mt, At)

    return im_fov.to(torch.uint8)


class FoveateImage(torch.nn.Module):
    """
    This class is a torchvision transform module that foveates an image with given fixations.
    """

    def forward(self, img: torch.Tensor, fixs: list) -> torch.Tensor:
        return foveate_image(img, fixs)
