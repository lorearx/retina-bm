import matplotlib.pyplot as plt
from utils.crossing_number import calculate_minutiaes
import os
from click_spinner import spinner
from termcolor import colored
from feature_extraction import call_processing, call_template_computing
from recognition import execute_recognition

FOLDER = "samples_for_enrollment"
RECOGN_FOLDER = "samples_for_recognition"
DATABASE_FOLDER = "retina_database"

def run_demo():
    text = colored("[RUNNING DEMO MODE - ENROLLMENT]", "light_magenta", attrs=["reverse", "blink"])
    print(text)
    individuals = os.listdir(FOLDER)
    print(f"The invidivuals with sample to be processed are:")
    for i, ind in enumerate(individuals):
        print(f"\t[{i+1}] {ind}")
    selected = int(input("Please, select one: ")) - 1
    individual_selected = individuals[selected]
    path_to_individual = os.path.join(FOLDER, individual_selected)
    samples = os.listdir(path_to_individual)
    for index_sample in (range(len(samples))):
        print("Processing sample...", end='')
        with spinner(True):
            path_to_sample = os.path.join(path_to_individual, samples[index_sample])
            image_processed, optic_disk_center = call_processing(path_to_sample, verbose=True)
            minutiaes, _, _ = calculate_minutiaes(image_processed.astype('uint8'))
            plt.imshow(minutiaes)
            print("Showing minutiaes computed...")
            plt.show()
            if (not optic_disk_center is None):
                template = call_template_computing(image_processed, optic_disk_center, individual_selected, saving=False, verbose=True)
                if (template is None):
                    print()
                    text = colored(f"[DEMO] Invidiual could not be registered: unable to generate a code with center {optic_disk_center}", "red", attrs=["reverse", "blink"])
                    print(text)
            else:
                print()
                text = colored("[DEMO] There's a problem with the sample provided (no optic disk)", "red", attrs=["reverse", "blink"])
                print(text)
        print()
    
    text = colored("[RUNNING DEMO MODE - RECOGNITION]", "light_magenta", attrs=["reverse", "blink"])
    print(text)

    path_to_individual = os.path.join(RECOGN_FOLDER, individual_selected)
    samples = os.listdir(path_to_individual)
    num_samples = len(samples)
    for index_sample in range(num_samples):
        execute_recognition(path_to_individual, samples, index_sample, individual_selected, verbose=True)