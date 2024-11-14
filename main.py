from DELG_datasets import ImageFolderCustomV2
from torch.utils.data import DataLoader

def main():
    root_folder = ""
    # Dataset
    image_dataset = ImageFolderCustomV2(root_folder)
    # Dataloader
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True)

    # Model

    # Training or inference 


if __name__ == "__main__":
    main()