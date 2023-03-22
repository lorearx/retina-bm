import os
from click_spinner import spinner
from termcolor import colored
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
    num_samples = len(samples)
    if (num_samples==1):
        index_sample = 0
    else:
        index_sample = int(input(f"There are {num_samples} samples for {individual_selected}. Select the index of the one selected: ")) - 1
    print("Processing sample...", end='')
    with spinner(True):
        path_to_sample = os.path.join(path_to_individual, samples[index_sample])
        image_processed, optic_disk_center = call_processing(path_to_sample)
        if (not optic_disk_center is None):
            template = call_template_computing(image_processed, optic_disk_center, individual_selected, index_sample)
            if (template is None):
                print()
                text = colored(f"Invidiual could not be registered: unable to generate a code with center {optic_disk_center}", "red", attrs=["reverse", "blink"])
                print(text)
            else:
                text = colored(f"INDIVIDUAL {individual_selected} REGISTERED", "green", attrs=["reverse", "blink"])
                print()
                print(text)
        else:
            print()
            text = colored("There's a problem with the sample provided", "red", attrs=["reverse", "blink"])
            print(text)
    print()