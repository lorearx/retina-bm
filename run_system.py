from enrollment import run_enrollment
from recognition import run_recognition
import sys
from pyfiglet import Figlet

ENROLLMENT_MODE = "1"
RECOGNITION_MODE = "2"
EXIT_MODE = "exit"

MODES = [ENROLLMENT_MODE, RECOGNITION_MODE, EXIT_MODE]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run_mode(mode):
    if (mode == ENROLLMENT_MODE):
        run_enrollment()
    elif (mode == RECOGNITION_MODE):
        run_recognition()

f = Figlet(font='slant')
print(f.renderText('RETINA-BM'))

valid_mode = False
exit = False

while (not exit):
    mode_selected = input(f"This system has two modes:\n\t[1] {bcolors.OKBLUE}STAGE 1 (Creation){bcolors.ENDC}\n\t[2] {bcolors.OKCYAN}STAGE 2 (Recognition){bcolors.ENDC}\nPlease, select one: ")
    valid_mode = mode_selected in MODES
    exit = mode_selected == EXIT_MODE
    if (not exit):
        if valid_mode:
            run_mode(mode_selected)
        else:
            print(f"{bcolors.FAIL} Mode {mode_selected} does not exist. Select a valid one. {bcolors.ENDC}")
