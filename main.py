from utils import aria_algorithm_segmentation
import cv2 as cv
import matplotlib.pyplot as plt
from utils.aria_utils.vessel_data import Vessel_Data
from utils.crossing_number import calculate_minutiaes

# Segmentate vessel
image = cv.imread("RIDB\P14\IM000001_14.JPG", 0) # Grayscale
#image = cv.resize(image, (500,500))

vessel_data = Vessel_Data()
vessel_data.im = image
vessel_data.bw_mask = None
vessel_data.bw = image
args = {
    'mask_option': "create",
    'mask_dark_threshold': 10,
    "mask_bright_threshold" : 255,
    "mask_largest_region": True,
    "only_thinning": True
}
aria_algorithm_segmentation.aria_vessel_segmentation(vessel_data, args)

# vessel_data.show()

thinned = vessel_data.thin
if (not thinned is None):
    plt.imshow(calculate_minutiaes(thinned.astype('uint8')))
    plt.show()
