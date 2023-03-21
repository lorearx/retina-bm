import os
from feature_extraction import call_processing, call_template_computing
from termcolor import colored
import numpy as np
from scipy.stats import pearsonr
from click_spinner import spinner
from dtw import dtw

FOLDER = "samples_for_recognition"
DATABASE_FOLDER = "retina_database"

def compute_metric(vector1, vector2):
    v1 = vector1 / np.linalg.norm(vector1)
    v2 = vector2 / np.linalg.norm(vector2)
    #dist, cost, acc, path = dtw(v1, v2, dist=lambda x, y: np.abs(x - y))
    corr, p_value = pearsonr(v1, v2)
    print(f"Similarity: {corr:.5f}")

def find_match(template):
    match = None
    templates = os.listdir(DATABASE_FOLDER)
    for template_name in templates:
        saved_template = np.load(os.path.join(DATABASE_FOLDER, template_name))
        compute_metric(template, saved_template)
    return match

def run_recognition():
    text = colored("[RUNNING RECOGNITION MODE]", "light_magenta", attrs=["reverse", "blink"])
    print(text)
    individuals = os.listdir(FOLDER)
    print(f"The invidivuals to be recognised are: {individuals}")
    selected = int(input("Please, select one:")) - 1
    individual_selected = individuals[selected]
    path_to_individual = os.path.join(FOLDER, individual_selected)
    samples = os.listdir(path_to_individual)

    # TODO: selecting sample
    print("Processing sample...", end='')
    with spinner():
        path_to_sample = os.path.join(path_to_individual, samples[0])
        image_processed = call_processing(path_to_sample)
        template = call_template_computing(image_processed, individual_selected, saving=False)
        if (not template is None):
            match = find_match(template)
            if (not match is None):
                text = colored("INDIVIDUAL REGISTERED", "green", attrs=["reverse", "blink"])
                print(text)
            else:
                text = colored("UNKNOWN INDIVIDUAL", "red", attrs=["reverse", "blink"])
                print(text)