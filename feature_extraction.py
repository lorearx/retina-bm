from utils import aria_algorithm_segmentation
import cv2
import matplotlib.pyplot as plt
from utils.aria_utils.vessel_data import Vessel_Data
import numpy as np
from template import compute_template, save_template

def get_optic_disk_center(sample_selected):
    optic_disk_center = None
    bright_pixels = (sample_selected>(np.max(sample_selected)*0.80))*255
    circles = cv2.HoughCircles(bright_pixels.astype('uint8'), cv2.HOUGH_GRADIENT,1,40, param1=50,param2=5,minRadius=30,maxRadius=70)
    if (not circles is None):
        if (len(circles)==1):
            circles = np.uint16(np.around(circles))
            i = circles[0,:][0]
            optic_disk_center = (i[1],i[0])
    return optic_disk_center

def call_processing(path_to_sample, verbose=False):
    sample_selected = cv2.imread(path_to_sample, 0)
    vessel_data = Vessel_Data()
    vessel_data.im = sample_selected
    vessel_data.bw_mask = None
    vessel_data.bw = sample_selected
    args = {
        'mask_option': "create",
        'mask_dark_threshold': 10,
        "mask_bright_threshold" : 255,
        "mask_largest_region": True,
        "only_thinning": True
    }
    aria_algorithm_segmentation.aria_vessel_segmentation(vessel_data, args)
    optic_disk_center = get_optic_disk_center(sample_selected)

    if (verbose):
        plt.imshow(vessel_data.thin)
        print(f"Showing skeleton with optic disk center located at {optic_disk_center}...")
        plt.show()
    return vessel_data.thin, optic_disk_center


def call_template_computing(image, optic_disk_center, individual, index_sample = None, saving=True, verbose=False):
    template = None
    if (not image is None):
        template = compute_template(image, optic_disk_center)
        if (saving):
            save_template(template, individual, index_sample)
        if (verbose):
            plt.imshow(template)
            print("Showing retina code...")
            plt.show()
    return template