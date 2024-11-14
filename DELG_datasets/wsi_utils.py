from constants.path_openslide_window import OPENSLIDE_PATH
import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
    
from PIL import Image
import matplotlib.pyplot as plt 


class WholeSlideImage:
    def __init__(self, path):
        self.path = path
        self.os_obj = openslide.OpenSlide(path)
        self.dimensions = self.os_obj.dimensions
        self.level_count = self.os_obj.level_count
        self.level_dimensions = self.os_obj.level_dimensions
        self.level_downsamples = self.os_obj.level_downsamples

    def read_region(self, location, level, size):
        return self.os_obj.read_region(location, level, size)

    def get_thumbnail(self):
        return self.os_obj.get_thumbnail(self.os_obj.dimensions)

    def get_level_downsamples(self):
        return self.level_downsamples

    def get_level_dimensions(self):
        return self.level_dimensions

    def get_level_count(self):
        return self.level_count

    def get_dimensions(self):
        return self.dimensions

    def get_os_obj(self):
        return self.os_obj

    def show(self, downsample_level):
        img = self.os_obj.read_region((0, 0), downsample_level, self.os_obj.level_dimensions[downsample_level])
        img = img.convert("RGB")
        plt.imshow(img)
        plt.show()
        
    def close(self):
        self.os_obj.close()

    def convert_to_jpeg2000(self, output_file, level, quality):
        image = self.os_obj.read_region((0, 0), level, self.os_obj.level_dimensions[level])
        image = image.convert("RGB")
        image.save(output_file, "JPEG2000", quality_mode="rates", quality_layers=[quality])
        

        
if __name__ == "__main__":
    file_svs_path = "/Volumes/Untitled 2 1/3-CNN-Tensorflow/26-Pytorch/15-image-similarity-search/fish/TCGA-jp2/TCGA-GBM/TCGA-06-0125-01A-01-BS1.jp2"
    wsi = WholeSlideImage(file_svs_path)
    wsi.show(file_svs_path)
    print(wsi.get_level_downsamples())
    print(wsi.get_level_dimensions())
    print(wsi.get_level_count())
    print(wsi.get_dimensions())
    
    # file_svs_basename = os.path.basename(file_svs_path)
    # output_file = file_svs_basename.split(".")[0] + ".jp2"
    # wsi.convert_to_jpeg2000(output_file, level=1, quality=50)
    
    wsi.close()