import argparse
import cv2
import h5py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from constants.path_openslide_window import OPENSLIDE_PATH
import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def save_h5_patches_after_remove_artifacts(image_slide_path, coord_path, patch_size, save_patch_dir, threshold_white=0.9):
    """
    Save patches after removing artifacts
    Input:
        image_slide_path (str): The path to the slide image
        coord_path (str): The path to the HDF5 file containing the coordinates of patches
        patch_size (int): The patch size used to patch the slide
        save_patch_dir (str): The directory to save the patches
    """
    wsi = openslide.open_slide(image_slide_path)
    with h5py.File(coord_path, 'r') as hf:
        coords = hf['coords'][:]
    
    clean_coords = []
    for coord in coords:
        region = wsi.read_region(coord, 0, (patch_size, patch_size)).convert("L").resize((256, 256))
        _, white_region = cv2.threshold(np.array(region), 235, 255, cv2.THRESH_BINARY)
        if np.sum(white_region == 255) / (256 * 256) <= threshold_white:
            clean_coords.append(coord)

    image_basename = os.path.basename(image_slide_path)
    os.makedirs(save_patch_dir, exist_ok=True)
    os.makedirs(os.path.join(save_patch_dir, image_basename), exist_ok=True)
    
    folder_save_patches_image_path = os.path.join(save_patch_dir, image_basename)
    for idx, coord in enumerate(clean_coords):
        patch = wsi.read_region(coord, 0, (patch_size, patch_size))
        patch_image = patch.convert("RGB")
        patch_save_path = os.path.join(folder_save_patches_image_path, f'patch_{idx}.png')
        patch_image.save(patch_save_path)

def arfifacts_removal(folder_slides_path:str, folder_mosaic_patches_path:str, patch_size, folder_save_clean_patch_dir):
    """
    Remove artifacts from the patches
    Input:
        folder_slides_path (str): The path to the slide images
        folder_mosaic_patches_path (str): The path to the patches
        patch_size (int): The patch size used to patch the slide
        folder_save_clean_patch_dir (str): The directory to save the clean patches
    """
    os.makedirs(folder_save_clean_patch_dir, exist_ok=True)
    image_slide_paths = [os.path.join(folder_slides_path, image_name) for image_name in os.listdir(folder_slides_path)]
    coord_paths = [os.path.join(folder_mosaic_patches_path, coord_name) for coord_name in os.listdir(folder_mosaic_patches_path)]

    for image_slide_path, coord_path in zip(image_slide_paths, coord_paths):
        save_h5_patches_after_remove_artifacts(image_slide_path, coord_path, patch_size, folder_save_clean_patch_dir)

