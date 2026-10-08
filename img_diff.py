import numpy as np
from PIL import Image


def main():
    with Image.open("data/Training/pituitary/Tr-pi_1155.jpg") as img_rgb:
        pixels1 = np.asarray(img_rgb)

    with Image.open("data/Training/pituitary/Tr-pi_766.jpg") as img_gs:
        pixels2 = np.asarray(img_gs)

    print(pixels1.shape)
    print(pixels2.shape)

    red = pixels1[: , : , 0]
    green = pixels1[: , : , 1]
    blue = pixels1[: , : , 2]

    x = [red, green, blue]

    for item in x:
        print("Color:")
        print(item)

    diff = red - green
    print(diff.max())
    print(np.array_equal(red, green))
    print(np.array_equal(green, blue))


if __name__ == "__main__":
    main()
