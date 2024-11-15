
# DELG-Search: TCGA Image Similarity Search

This project implements a system for exploring and visualizing images from the TCGA dataset. The main functionality allows you to view a set of CT images by TCGA type or randomly display images from the dataset.

## Table of Contents

- [Installation](#installation)
- [Preprocessing data](#Preprocessing-data)
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

2. Create virtual environment
    ```bash
    python3 -m venv venv
    ```
3. Install the required dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
4. Download the TCGA-jp2 dataset:
    ```bash
    scripts/download_dataset.sh
    ```

## Preprocessing-data all stages
    ```bash
    python3 pipeline_preprocessing.py --source_dir TCGA-jp2  --save_dir RESULT_PATCHES_DIRECTORY
    ```

## Preprocessing-data stage patches
    ```bash
    python3 pipeline_preprocessing.py --source_dir TCGA-jp2  --save_dir RESULT_PATCHES_DIRECTORY --upload_drive --stage "patches"
    ```
## Preprocessing-data stage mosaic
    ```bash
    python3 pipeline_preprocessing.py --source_dir TCGA-jp2  --save_dir RESULT_PATCHES_DIRECTORY --upload_drive --stage "mosaic"
    ```
## Preprocessing-data stage clean
    ```bash
    python3 pipeline_preprocessing.py --source_dir TCGA-jp2  --save_dir RESULT_PATCHES_DIRECTORY --upload_drive --stage "clean"
    ```


## Dataset Structure

The dataset TCGA (CT images) should be organized in a folder structure where each TCGA type (such as KIRC, LUAD, etc.) has its own directory. Each directory contains images, typically in `.jp2` format.

Example structure before preprocessing:

```
/path/to/TCGA-jp2/
    ├── TCGA-KIRC/
    │   ├── image1.jp2
    │   ├── image2.jp2
    │   └── ...
    └── TCGA-LUAD/
        ├── image1.jp2
        ├── image2.jp2
        └── ...
```

Example structure after preprocessing:

```
/path/to/RESULT_PATCHES_DIRECTORY/
    ├── TCGA-KIRC/
    ├   ├── clean_patches # final result after processing
    │   ├── masks
    │   ├── patches
    │   └── stitches
    └── TCGA-LUAD/
    |   ├── clean_patches
    │   ├── masks
    │   ├── patches
    │   └── stitches
```

## NOTE: IF USE WINDOW
- Download this file: https://drive.google.com/file/d/1pgwxTOeduKLWYlkCPcV3CYUrsW2xgZ3i/view?usp=drive_link
- Then extract at anywhere you want
- Finally, change the folder directory path in constants/path_openslide_window.py
```
    OPENSLIDE_PATH = r'C:\Users\Le Quang Nhat\Downloads\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin'  # Replace 
```