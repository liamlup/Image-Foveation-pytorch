from foveate import FoveateImage
from torchvision.io import read_image
import matplotlib.pyplot as plt
from random import randint

if __name__ == '__main__':
    img = read_image("data/face.jpg")
    img_batch = img.repeat(16, 1, 1, 1)
    xc, yc = int(img.shape[2] / 2), int(img.shape[1] / 2)
    fixations = [(randint(0, img.shape[2]), randint(0, img.shape[1])) for _ in range(16)]

    foveate = FoveateImage()(img_batch, fixations)

    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(foveate[i * 4 + j].permute(1, 2, 0))
            ax[i, j].scatter(fixations[i * 4 + j][0], fixations[i * 4 + j][1], c='r', s=30, marker='x')
    plt.show()