
# DELG-Search: TCGA Image Similarity Search

This project implements a system for exploring and visualizing images from the TCGA dataset. The main functionality allows you to view a set of CT images by TCGA type or randomly display images from the dataset.

## Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Arguments](#arguments)
- [License](#license)

## Installation

To set up this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/nate-q-136/DELG-Search.git
    cd DELG-Search
    ```

2. Install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    Ensure that the Python environment is using versions compatible with the libraries listed in `requirements.txt`.

3. If you don't have the dataset, run the provided script to download it:
    ```
    chmod +x scripts/download_dataset.sh
    ```
    ```bash
    sh scripts/download_dataset.sh
    ```

## Dataset Structure

The dataset TCGA (CT images) should be organized in a folder structure where each TCGA type (such as KIRC, LUAD, etc.) has its own directory. Each directory contains images, typically in `.jpg` format.

Example structure:

```
/path/to/tcga_dataset/
    ├── TCGA-KIRC/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── TCGA-LUAD/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

Ensure your dataset follows this structure to use the project correctly.

## Usage

You can run the program to visualize the images from the dataset. The program supports two main modes:
- Display images by **TCGA type**.
- Display random images from **the dataset**.

### Display images by TCGA type

To display images from a specific TCGA type, use the following command:

```bash
python3 main_datasets.py --folder_base_path /path/to/dataset --tcga_type TCGA-KIRC --show_by_type --top_k 5
```

This will display 5 images from the TCGA type `KIRC`. You can adjust the `--top_k` argument to display more or fewer images.

### Display random images from the dataset

To display random images from the dataset, use the following command:

```bash
python3 main_datasets.py --folder_base_path /path/to/dataset --show_by_dataset --top_k 5
```

This will display 5 random images from different classes (TCGA types) in the dataset.

### Arguments

- `--folder_base_path`: Path to the base folder containing TCGA image data. (Required)
- `--tcga_type`: The TCGA type (e.g., KIRC, LUAD) to display images from. (Required when `--show_by_type` is used)
- `--show_by_type`: Flag to indicate if you want to display images by TCGA type.
- `--show_by_dataset`: Flag to indicate if you want to display random images from the dataset.
- `--top_k`: Number of images to display. Default is 5.

### Example

To display 3 random images from the dataset located in `/path/to/dataset`, you would run:

```bash
python main_datasets.py --folder_base_path /path/to/dataset --show_by_dataset --top_k 3
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
