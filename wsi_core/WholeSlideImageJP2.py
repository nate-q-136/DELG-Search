import math
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import h5py
from wsi_core.wsi_utils import (
    savePatchIter_bag_hdf5,
    initialize_hdf5_bag,
    coord_generator,
    save_hdf5,
    sample_indices,
    screen_coords,
    isBlackPatch,
    isWhitePatch,
    to_percentiles
)
from wsi_core.util_classes import (
    isInContourV1,
    isInContourV2,
    isInContourV3_Easy,
    isInContourV3_Hard,
    Contour_Checking_fn
)
from utils.file_utils import load_pkl, save_pkl

Image.MAX_IMAGE_PIXELS = None  # Remove limit for large images

class WholeSlideImage(object):
    def __init__(self, path, xml_path=None, hdf5_file=None):
        self.name = os.path.splitext(os.path.basename(path))[0]
        # Load the .jp2 image using OpenCV
        self.wsi = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.wsi is None:
            raise ValueError(f"Could not read the image file {path}")
        # Convert BGR to RGB
        self.wsi = cv2.cvtColor(self.wsi, cv2.COLOR_BGR2RGB)
        self.level_dim = [ (self.wsi.shape[1], self.wsi.shape[0]) ]  # [(width, height)]
        self.level_downsamples = [(1,1)]  # No downsampling levels in standard images
        self.contours_tissue = None
        self.contours_tumor = None
        self.seg_level = 0
        self.hdf5_file = hdf5_file

    def getOpenSlide(self):
        # No openslide, return the image
        return self.wsi

    def initXML(self, xml_path):
        def _createContour(coord_list):
            return np.array([[
                int(float(coord.attributes['X'].value)),
                int(float(coord.attributes['Y'].value))
            ] for coord in coord_list], dtype='int32').reshape(-1, 1, 2)

        xmldoc = minidom.parse(xml_path)
        annotations = [
            anno.getElementsByTagName('Coordinate')
            for anno in xmldoc.getElementsByTagName('Annotation')
        ]
        self.contours_tumor = [_createContour(coord_list) for coord_list in annotations]
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initTxt(self, annot_path):
        def _create_contours_from_dict(annot):
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1, 1, 2)
                        all_cnts.append(contour)
                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1, 1, 2)
                        all_cnts.append(contour)
            return all_cnts

        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)
        self.contours_tumor = _create_contours_from_dict(annot)
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initSegmentation(self, mask_file):
        asset_dict = load_pkl(mask_file)
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    def segmentTissue(
        self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False,
        filter_params={'a_t': 100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]
    ):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        def _filter_contours(contours, hierarchy, filter_params):
            """
            Filter contours by area.
            """
            filtered = []
            hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
            all_holes = []
            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                a = cv2.contourArea(cont)
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if a > filter_params['a_t']:
                    filtered.append(cont_idx)
                    all_holes.append(holes)
            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            hole_contours = []
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfiltered_holes = unfiltered_holes[:filter_params.get('max_n_holes', 10)]
                filtered_holes = []
                for hole in unfiltered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)
                hole_contours.append(filtered_holes)
            return foreground_contours, hole_contours

        img = self.wsi.copy()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
        if use_otsu:
            _, img_thresh = cv2.threshold(
                img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
            )
        else:
            _, img_thresh = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
        contours, hierarchy = cv2.findContours(
            img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if hierarchy is not None:
            hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
            if filter_params:
                foreground_contours, hole_contours = _filter_contours(
                    contours, hierarchy, filter_params
                )
                self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
                self.holes_tissue = self.scaleHolesDim(hole_contours, scale)
        else:
            self.contours_tissue = []
            self.holes_tissue = []

        if len(keep_ids) < 1:
            keep_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)
        self.contours_tissue = [self.contours_tissue[i] for i in keep_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in keep_ids]

    def visWSI(
        self, vis_level=0, color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0),
        line_thickness=12, max_size=None, top_left=None, bot_right=None, custom_downsample=1,
        view_slide_only=False, number_contours=False, seg_display=True, annot_display=True
    ):
        scale = [1 / self.level_downsamples[vis_level][0], 1 / self.level_downsamples[vis_level][1]]
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple(
                (np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int)
            )
            region_size = (w, h)
            img = self.wsi[
                int(top_left[1]):int(bot_right[1]), int(top_left[0]):int(bot_right[0]), :
            ]
        else:
            img = self.wsi.copy()
            region_size = img.shape[:2][::-1]  # width, height

        if not view_slide_only:
            offset = tuple(
                -(np.array(top_left) * scale).astype(int)
            ) if top_left is not None else (0, 0)
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(
                        img, self.scaleContourDim(self.contours_tissue, scale),
                        -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset
                    )
                else:
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        if M["m00"] == 0:
                            continue
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.drawContours(
                            img, [contour], -1, color, line_thickness,
                            lineType=cv2.LINE_8, offset=offset
                        )
                        cv2.putText(
                            img, "{}".format(idx), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10
                        )

                for holes in self.holes_tissue:
                    cv2.drawContours(
                        img, self.scaleContourDim(holes, scale),
                        -1, hole_color, line_thickness, lineType=cv2.LINE_8, offset=offset
                    )

            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(
                    img, self.scaleContourDim(self.contours_tumor, scale),
                    -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset
                )

        img_pil = Image.fromarray(img)
        w, h = img_pil.size
        if custom_downsample > 1:
            img_pil = img_pil.resize((int(w / custom_downsample), int(h / custom_downsample)))
        if max_size is not None and (w > max_size or h > max_size):
            resize_factor = max_size / w if w > h else max_size / h
            img_pil = img_pil.resize((int(w * resize_factor), int(h * resize_factor)))
        return img_pil

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(
                hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False
            ) > 0:
                return 1
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.level_dim[0]
        for downsample, dim in zip([1], self.level_dim):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append((downsample, downsample))
        return level_downsamples

    # You would need to adjust other methods similarly to handle .jp2 images loaded with OpenCV.

    # For example, methods like `createPatches_bag_hdf5`, `process_contours`, `visHeatmap`, etc.,
    # need to be adapted to work with the OpenCV image and consider that there are no multiple levels.

if __name__ == "__main__":
    # Example usage
    wsi_path = '/Volumes/Untitled 2 1/3-CNN-Tensorflow/26-Pytorch/15-image-similarity-search/fish/DELG/root_jp2/TCGA-A3-3362-01A-01-BS1.jp2'
    wsi_object = WholeSlideImage(wsi_path)
    seg_params = {
    'seg_level': 0,          # Since we have only one level
    'sthresh': 20,
    'sthresh_up': 255,
    'mthresh': 7,
    'close': 0,
    'use_otsu': False,
    'filter_params': {'a_t': 100, 'a_h': 16, 'max_n_holes': 10},
    'ref_patch_size': 512,
    'exclude_ids': [],
    'keep_ids': []
    }
    vis_params = {
        'vis_level': 0,
        'color': (0, 255, 0),        # Color for tissue contours (Green)
        'hole_color': (0, 0, 255),   # Color for holes (Red)
        'annot_color': (255, 0, 0),  # Color for annotations (Blue)
        'line_thickness': 2,
        'max_size': None,
        'top_left': None,
        'bot_right': None,
        'custom_downsample': 1,
        'view_slide_only': False,
        'number_contours': False,
        'seg_display': True,
        'annot_display': False
    }

    wsi_object.segmentTissue(**seg_params)
    img_pil = wsi_object.visWSI(**vis_params)
    img_pil.show()

