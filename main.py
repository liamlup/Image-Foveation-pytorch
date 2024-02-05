from foveate import FoveateImage
from torchvision.io import read_image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = read_image("data/face.jpg")
    xc, yc = int(img.shape[2]/2), int(img.shape[1]/2)
    foveate = FoveateImage()(img, [(xc, yc)])
    plt.imshow(foveate.permute(1, 2, 0))
    plt.scatter(xc, yc)
    plt.show()
