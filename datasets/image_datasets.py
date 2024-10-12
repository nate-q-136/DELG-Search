from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import os
import pathlib
import torch
from PIL import Image
import matplotlib.pyplot as plt


# make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Returns all class names in a given directory and the corresponding class index
    """
    class_names = [
        entry.name for entry in list(os.scandir(directory)) if entry.is_dir()
    ]
    class_index = {class_name: index for index, class_name in enumerate(class_names)}
    return class_names, class_index


def display_pil_image(ax, image_path):
    image = Image.open(image_path)
    ax.imshow(image, cmap="gray")
    ax.axis("off")  # Hide axis for a cleaner look


def display_type_tcga_images(folder_base_path, tcga_type, number_images=5):
    # Path to the folder containing the converted JPEG images
    folder_path = f"{folder_base_path}/{tcga_type}"
    image_files = os.listdir(folder_path)

    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Sample Images from {tcga_type}", fontsize=16)

    # Loop to display images in subplots
    for i in range(number_images):
        image_path = os.path.join(folder_path, image_files[i])
        ax = plt.subplot(1, number_images, i + 1)
        display_pil_image(ax, image_path)
        ax.set_title(f"Image {i + 1}")

    # Display the entire figure with subplots
    plt.tight_layout()
    plt.show()


# 1. Create a custom dataset derived from torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    # 2. Initialize with a target_dir and tranform (optional) parameter
    def __init__(self, target_dir: str, transform=None) -> None:
        # 3. create class attributes
        # get all image paths
        self.paths = list(pathlib.Path(target_dir).glob("*/*.jpg"))
        # setup transform
        self.transform = transform
        # create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(target_dir)

    # 4. Make a function to load images
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method
    def __len__(self) -> int:
        return len(self.paths)

    # 6. Overwrite the __getitem__() method
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return one sample of data (X, y)"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name  # Take the parent of the image path
        class_idx = self.class_to_idx[class_name]
        # transform the image if necessary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
