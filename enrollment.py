from utils import aria_algorithm_segmentation
import cv2
import matplotlib.pyplot as plt
from utils.aria_utils.vessel_data import Vessel_Data
import os
import numpy as np
from click_spinner import spinner
from termcolor import colored, cprint
from template import compute_template, save_template
from feature_extraction import call_processing, call_template_computing

FOLDER = "samples_for_enrollment"

def run_enrollment():
    text = colored("[RUNNING ENROLLMENT MODE]", "light_magenta", attrs=["reverse", "blink"])
    print(text)
    individuals = os.listdir(FOLDER)
    print(f"The invidivuals ready for enrollment are:")
    for i, ind in enumerate(individuals):
        print(f"\t[{i+1}] {ind}")
    selected = int(input("Please, select one: ")) - 1
    individual_selected = individuals[selected]
    path_to_individual = os.path.join(FOLDER, individual_selected)
    samples = os.listdir(path_to_individual)
    if len(samples) == 1:
        print("Processing sample...", end='')
        with spinner(True):
            path_to_sample = os.path.join(path_to_individual, samples[0])
            image_processed = call_processing(path_to_sample)
            call_template_computing(image_processed, individual_selected)
        print()