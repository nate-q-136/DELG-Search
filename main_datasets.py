from matplotlib import pyplot as plt
from datasets import ImageFolderCustom, display_type_tcga_images, find_classes
from argparse import ArgumentParser
import random

def main():
    # Initialize ArgumentParser
    parser = ArgumentParser(description="TCGA Image Folder Arguments")
    
    # Add folder_base_path and tcga_type arguments
    parser.add_argument('--folder_base_path', type=str, help="Path to the base folder containing TCGA image data")
    parser.add_argument('--tcga_type', type=str, help="TCGA type (e.g., KIRC, LUAD, etc.)", default="")
    
    # Add --show flag to control displaying images
    parser.add_argument('--show_by_type', action='store_true', help="Display images if set")
    parser.add_argument('--show_by_dataset', action='store_true', help="Display images from dataset if set")
    # Add top_k argument to limit number of images to show if --show is set
    parser.add_argument('--top_k', type=int, default=5, help="Number of images to show if --show is set (default: 5)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Display images if --show is set
    if args.show_by_type:
        display_type_tcga_images(args.folder_base_path, args.tcga_type, number_images=args.top_k)
    elif args.show_by_dataset:
        transform = None
        image_dataset = ImageFolderCustom(args.folder_base_path, transform=transform)
        print(f"Classes: {image_dataset.classes}")
        print(f"Class to index: {image_dataset.class_to_idx}")

        # Display images from the dataset
        plt.figure(figsize=(15, 5))
        plt.suptitle("Sample Images from Dataset", fontsize=16)
        for i in range(args.top_k):
            index_random = random.randint(0, len(image_dataset) - 1)
            image, label = image_dataset[index_random]
            ax = plt.subplot(1, args.top_k, i + 1)
            if transform:
                image = image.permute(1, 2, 0)
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Class: {image_dataset.classes[label]}")
        plt.tight_layout()
        plt.show()
    else:
        print("Flag --show_by_type or --show_by_dataset is not set, skipping image display.")


if __name__ == "__main__":
    main()