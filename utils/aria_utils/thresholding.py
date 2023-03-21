import numpy as np
from scipy import ndimage

def get_largest_region(bw):
    # Identify connected components
    cc = ndimage.label(bw)
    
    # Initialise output
    bw_largest = np.zeros_like(bw, dtype=bool)

    label_im, nb_labels = ndimage.label(bw)
    # No regions found
    if cc[1] >= 1:
        sizes = ndimage.sum(bw, label_im, range(nb_labels + 1))
        mask = sizes == max(sizes)
        bw_largest = mask[label_im]
    
    return bw_largest

def mask_threshold(im, dark_threshold, bright_threshold, largest_region):
    # Apply the thresholds
    bw = np.logical_and(im >= dark_threshold, im <= bright_threshold)
    
    # Determine the single largest region
    if largest_region:
        bw = get_largest_region(bw)
    
    return bw