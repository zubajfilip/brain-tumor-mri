from collections import Counter
from pathlib import Path

from PIL import Image

TRAINING_DIR = Path("data/Training")
TESTING_DIR = Path("data/Testing")


def count_data(folder: Path) -> dict[str, int]:
    """Count .jpg images in each class subfolder of `folder`."""
    counts_per_class = {}
    for item in folder.iterdir():
        if item.is_dir():
            image_paths = item.glob("*.jpg")
            count = len(list(image_paths))
            counts_per_class[item.name] = count
    return counts_per_class


def get_image_stats(folder: Path) -> tuple[Counter[tuple[int, int]], Counter[str]]:
    """Get image sizes and modes for all .jpg images in `folder`."""
    sizes = Counter()
    modes = Counter()
    for image_path in folder.rglob("*.jpg"):
        with Image.open(image_path) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1

    return (sizes, modes)



def main():
    for folder in [TRAINING_DIR, TESTING_DIR]:
        counts = count_data(folder)
        print(f"Counts for {folder}: {counts}")

    sizes_train, modes_train = get_image_stats(Path("data/Training"))
    sizes_test, modes_test = get_image_stats(Path("data/Testing"))

    print("====================TRAIN DATA==========================")
    print("SIZES")
    for key, value in sizes_train.most_common(5):
        print(f"Size: {key}, Occurs: {value}")
    print(f"Total: {len(sizes_train)}")

    print("MODES")
    for key, value in modes_train.most_common(5):
        print(f"Mode: {key}, Occurs: {value}")
    print(f"Total: {len(modes_train)}")

    print("====================TEST DATA==========================")
    print("SIZES")
    for key, value in sizes_test.most_common(5):
        print(f"Size: {key}, Occurs: {value}")
    print(f"Total: {len(sizes_test)}")

    print("MODES")
    for key, value in modes_test.most_common(5):
        print(f"Mode: {key}, Occurs: {value}")
    print(f"Total: {len(modes_test)}")


if __name__ == "__main__":
    main()
