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


def print_image_stats(name: str, sizes: Counter, modes: Counter) -> None:
    """Print the most common image sizes and modes of one dataset split."""
    print(f"===================={name.upper()} DATA==========================")
    print("SIZES")

    for key, value in sizes.most_common(5):
        print(f"Size: {key}, Occurs: {value}")
    print(f"Distinct sizes: {len(sizes)}")

    print("MODES")

    for key, value in modes.most_common(5):
        print(f"Mode: {key}, Occurs: {value}")
    print(f"Distinct modes: {len(modes)}")


def main() -> None:
    for folder in [TRAINING_DIR, TESTING_DIR]:
        counts = count_data(folder)
        print(f"Counts for {folder}: {counts}")

        sizes, modes = get_image_stats(folder)
        print_image_stats(folder.name, sizes, modes)


if __name__ == "__main__":
    main()
