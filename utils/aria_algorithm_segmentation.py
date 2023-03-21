from utils.aria_utils import segmentation
from utils.aria_utils import thresholding
from utils.aria_utils import spline_fit
from utils.aria_utils import edges_gradient

def mask_selection(vessel_data, args):
    mask_option = args['mask_option']
    if (mask_option=='read_file'):
        mask = None
    elif (mask_option=='create'):
        mask_dark_threshold = args["mask_dark_threshold"]
        mask_bright_threshold = args["mask_bright_threshold"]
        mask_largest_region = args["mask_largest_region"]
        mask = thresholding.mask_threshold(vessel_data.im, mask_dark_threshold, mask_bright_threshold, mask_largest_region)
    else:
        mask = None
    return mask

def aria_vessel_segmentation(vessel_data, args):

    # If there isn't a mask there already, choose whether to apply one
    bw_mask = vessel_data.bw_mask
    if (bw_mask is None):
        vessel_data.bw_mask = mask_selection(vessel_data, args)

    # Segment the image using the isotropic undecimated wavelet transform
    segmentation.seg_iuwt(vessel_data)

    # Compute centre lines and profiles by spline-fitting
    spline_fit.centre_spline_fit(vessel_data, args)

    # Do the rest of the processing, and detect vessel edges using a gradient
    #edges_gradient.edges_max_gradient(vessel_data, args=None)