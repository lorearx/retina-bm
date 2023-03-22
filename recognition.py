import os
from feature_extraction import call_processing, call_template_computing
from termcolor import colored
import numpy as np
from click_spinner import spinner

FOLDER = "samples_for_recognition"
DATABASE_FOLDER = "retina_database"

def compute_metric(x, y):
    xb = np.unpackbits(np.array(x, dtype=np.uint8))
    yb = np.unpackbits(np.array(y, dtype=np.uint8))
    hamming_distance = np.count_nonzero(xb != yb)
    return hamming_distance

def find_match(template, verbose=False):
    print()
    match = None
    templates = os.listdir(DATABASE_FOLDER)
    for template_name in templates:
        saved_template = np.load(os.path.join(DATABASE_FOLDER, template_name))
        metric = compute_metric(template, saved_template)
        if (verbose):
            print(f"\tMetric obtained for {template_name}: {metric}")
        if (metric < 1000):
            match = template_name
            break
    return match

def execute_recognition(path_to_individual, samples, index_sample, individual_selected, verbose=False):
    print("Processing sample...", end='')
    with spinner():
        path_to_sample = os.path.join(path_to_individual, samples[index_sample])
        image_processed, optic_disk_center = call_processing(path_to_sample)
        if (not optic_disk_center is None):
            template = call_template_computing(image_processed, optic_disk_center, individual_selected, saving=False)
            if (not template is None):
                match = find_match(template, verbose)
                if (not match is None):
                    text = colored(f"INDIVIDUAL {individual_selected} KNOWN. ACCESS GRANTED TO {match}", "green", attrs=["reverse", "blink"])
                    print()
                    print(text)
                else:
                    text = colored(f"UNKNOWN INDIVIDUAL. ACCESS DENIED", "red", attrs=["reverse", "blink"])
                    print()
                    print(text)
            else:
                text = colored(f"There's a problem with the sample provided (no code generated with center {optic_disk_center})", "red", attrs=["reverse", "blink"])
                print()
                print(text)
        else:
            text = colored("There's a problem with the sample provided (no optic disk)", "red", attrs=["reverse", "blink"])
            print()
            print(text)

def run_recognition():
    text = colored("[RUNNING RECOGNITION MODE]", "light_magenta", attrs=["reverse", "blink"])
    print(text)
    individuals = os.listdir(FOLDER)
    print(f"The invidivuals to be recognised are:")
    for i, ind in enumerate(individuals):
        print(f"\t[{i+1}] {ind}")
    selected = int(input("Please, select one: ")) - 1
    individual_selected = individuals[selected]
    path_to_individual = os.path.join(FOLDER, individual_selected)
    samples = os.listdir(path_to_individual)
    num_samples = len(samples)
    for index_sample in range(num_samples):
        execute_recognition(path_to_individual, samples, index_sample, individual_selected)