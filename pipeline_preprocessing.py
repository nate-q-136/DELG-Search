
import argparse
import os
from feature_extractor.create_patches_fp import processing_patches
from feature_extractor.extract_mosaic import extract_mosaic
from feature_extractor.artifacts_removal import arfifacts_removal

def pipeline(source_dir, save_dir, patch_size=256, step_size=256, seg=True, patch=True, stitch=True, save_mask=True):
    os.makedirs(save_dir, exist_ok=True)
    for subdir in os.listdir(source_dir):
        disease_dir = os.path.join(source_dir, subdir)
        save_preprocessing_dir = os.path.join(save_dir, subdir)
        os.makedirs(save_preprocessing_dir, exist_ok=True)
        processing_patches(disease_dir, save_preprocessing_dir, patch_size=patch_size, step_size=step_size, seg=seg, patch=patch, stitch=stitch, save_mask=save_mask)
        extract_mosaic(disease_dir, save_preprocessing_dir+"/patches", save_preprocessing_dir)
        arfifacts_removal(folder_slides_path=disease_dir, folder_mosaic_patches_path=save_preprocessing_dir+"/feature_mosaic_patches", patch_size=patch_size, folder_save_clean_patch_dir=save_preprocessing_dir+"/clean_patches")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()
    pipeline(args.source_dir, args.save_dir)

if __name__ == "__main__":
    main()