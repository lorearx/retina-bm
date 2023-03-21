from utils import aria_algorithm_segmentation
import cv2
import matplotlib.pyplot as plt
from utils.aria_utils.vessel_data import Vessel_Data
import os
import numpy as np
from click_spinner import spinner
from termcolor import colored, cprint
from template import compute_template, save_template

def call_processing(path_to_sample):
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

    return vessel_data.thin

def call_template_computing(image, individual, saving=True):
    template = None
    if (not image is None):
        template = compute_template(image)
        if (saving):
            save_template(template, individual)
    return template