import numpy as np
from matplotlib import pyplot as plt
from stringart.utils.image import ImageWrapper

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path, inverted=False)
cmap = "grey"


def plot_preprocess(src: np.ndarray, dst: np.ndarray, title: str, fname: str) -> None:
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle(title)

    axs[0].set_title("Input Image")
    axs[0].imshow(src, cmap=cmap)
    axs[0].axis("off")

    axs[1].set_title("Output Image")
    axs[1].imshow(dst, cmap=cmap)
    axs[1].axis("off")

    fig.show()
    fig.savefig(fname)


def main():
    src = image.copy()

    dst = ImageWrapper.histogram_equalization(src)
    plot_preprocess(src, dst, "Histogram Equalization", "../../outputs/misc/histogram_equalization_preprocess.png")

    dst = ImageWrapper.grayscale_quantization(src)
    plot_preprocess(src, dst, "Grayscale Quantization", "../../outputs/misc/grayscale_quantization_preprocess.png")


if __name__ == "__main__":
    main()
