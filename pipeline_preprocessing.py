import argparse
import os
from feature_extractor.create_patches_fp import processing_patches
from feature_extractor.extract_mosaic import extract_mosaic
from feature_extractor.artifacts_removal import arfifacts_removal
from utils.ggdrive import upload_to_drive
import shutil
import time


def pipeline(
    source_dir,
    save_dir,
    patch_size=256,
    step_size=256,
    seg=True,
    patch=True,
    stitch=True,
    save_mask=True,
    upload_drive=False,
    stage="all",
):
    os.makedirs(save_dir, exist_ok=True)
    for subdir in os.listdir(source_dir):
        disease_dir = os.path.join(source_dir, subdir)
        save_preprocessing_dir = os.path.join(save_dir, subdir)
        os.makedirs(save_preprocessing_dir, exist_ok=True)
        if stage == "patches":
            processing_patches(
                disease_dir,
                save_preprocessing_dir,
                patch_size=patch_size,
                step_size=step_size,
                seg=seg,
                patch=patch,
                stitch=stitch,
                save_mask=save_mask,
            )
        elif stage == "mosaic":
            extract_mosaic(
                disease_dir, save_preprocessing_dir + "/patches", save_preprocessing_dir
            )
        elif stage == "clean":
            arfifacts_removal(
                folder_slides_path=disease_dir,
                folder_mosaic_patches_path=save_preprocessing_dir
                + "/feature_mosaic_patches",
                patch_size=patch_size,
                folder_save_clean_patch_dir=save_preprocessing_dir + "/clean_patches",
            )
        else:

            processing_patches(
                disease_dir,
                save_preprocessing_dir,
                patch_size=patch_size,
                step_size=step_size,
                seg=seg,
                patch=patch,
                stitch=stitch,
                save_mask=save_mask,
            )
            extract_mosaic(
                disease_dir, save_preprocessing_dir + "/patches", save_preprocessing_dir
            )
            arfifacts_removal(
                folder_slides_path=disease_dir,
                folder_mosaic_patches_path=save_preprocessing_dir
                + "/feature_mosaic_patches",
                patch_size=patch_size,
                folder_save_clean_patch_dir=save_preprocessing_dir + "/clean_patches",
            )

    if upload_drive:
        date_time = time.strftime("%Y%m%d-%H%M%S")
        # Zip the save_dir folder after processing
        zip_filename = f"{save_dir}_{date_time}.zip"
        zip_path = shutil.make_archive(zip_filename, "zip", save_dir)
        print(f"Folder zipped as {zip_filename}")

        # Upload the zip file to Google Drive
        upload_to_drive(zip_path, folder_id="1qJuckL9fUPFf51iHVPsht5A6T4Yq3KLI")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--upload_drive", action="store_true")
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--step_size", default=256, type=int)
    parser.add_argument(
        "--stage", default="all", choices=["patches", "mosaic", "clean", "all"]
    )
    args = parser.parse_args()
    pipeline(
        args.source_dir,
        args.save_dir,
        args.patch_size,
        args.step_size,
        upload_drive=args.upload_drive,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
