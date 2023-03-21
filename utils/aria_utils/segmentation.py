import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage

def numel(x):
    if isinstance(x, (list, tuple)):
        return len(x)
    elif isinstance(x, np.ndarray):
        return np.size(x)
    else:
        raise TypeError('Input must be a list, tuple, or NumPy array')
    
def dilate_wavelet_kernel(h, spacing):

    # Preallocate the expanded filter
    h2 = np.zeros(numel(h) + spacing * (numel(h) - 1))
    #h2 = h2.reshape(-1, 1)
    # Put in the coefficients
    h2[::spacing+1] = h
    return h2

def iuwt_vessels(im, levels, padding='same'):
    """
    Compute the sum of one or more wavelet levels computed using the
    isotropic undecimated wavelet transform (IUWT).
    
    Args:
    - im: the input image
    - levels: a 1-dimensional vector giving the wavelet levels to compute
    - padding: the same as the padding input argument in convolve2d; here it
      is 'symmetric' by default.
    
    Returns:
    - w: the sum of the requested wavelet levels
    
    This will compute transform and return the sum of the requested levels.
    Therefore
        w = iuwt_vessels(im, np.arange(1, 6))
    is equivalent to
        w = np.sum(iuwt_vessels_all(im, np.arange(1, 6)), axis=2)
    assuming that im is a 2D image, but using this function is more
    efficient because individual levels are not stored longer than necessary.
    """
    
    # First smoothing level = input image
    s_in = im
    
    # Inititalise output
    w = 0
    
    # B3 spline coefficients for filter
    b3 = np.array([1, 4, 6, 4, 1]) / 16
    
    # Compute transform
    for ii in range(1, levels[-1] + 1):
        # Create convolution kernel
        h = dilate_wavelet_kernel(b3, 2**(ii-1)-1)
        
        # Convolve and subtract to get wavelet level
        s_out = convolve2d(s_in, np.outer(h, h), mode=padding)
        
        # Add wavelet level if in levels
        if ii in levels:
            w += s_in - s_out
        
        # Update input for new iteration
        s_in = s_out
    
    return w

def percentage_threshold(data, proportion, sorted_data=False):
    """Determine a threshold so that (approx) proportion of data is above the 
    threshold.

    Args:
        data (array-like): The data from which the threshold should be computed
        proportion (float): The proportion of the data that should exceed the
            threshold.  If > 1, it will first be divided by 100.
        sorted_data (bool): True if the data has already been sorted, False otherwise

    Returns:
        tuple: (threshold, data_sorted)
        threshold (float): Either +Inf, -Inf or an actual value present in data.
        data_sorted (numpy.ndarray): A sorted version of the data, that might be used later to
            determine a different threshold.

    """
    # Need to make data a vector
    data = np.ravel(data)
    
    # If not told whether data is sorted, need to check
    if not sorted_data:
        sorted_data = np.all(np.diff(data) >= 0)
    
    # Sort data if necessary
    if not sorted_data:
        data_sorted = np.sort(data)
    else:
        data_sorted = data
    
    # Calculate threshold value
    if proportion > 1:
        proportion = proportion / 100
    proportion = 1 - proportion
    thresh_ind = round(proportion * data_sorted.size)
    if thresh_ind > data_sorted.size:
        threshold = np.inf
    elif thresh_ind < 1:
        threshold = -np.inf
    else:
        threshold = data_sorted[thresh_ind-1]
    
    return threshold, data_sorted

def percentage_segment(im, proportion, dark=True, bw_mask=None, sorted_pix=None):
    """
    Threshold an image to find a proportion of the darkest or brightest pixels.
    :param im: The image to threshold.
    :param proportion: The proportion of pixels to include. It should be a value between 0 and 1.
    :param dark: True if the lowest PERCENT pixels will be kept, otherwise the highest will be kept.
    :param bw_mask: (optional) A binary image corresponding to IM that gives the field of view to use.
    If absent, the entire image is used.
    :param sorted_pix: An array containing the pixels to use for computing the threshold.
    Sorting large numbers of pixels can be somewhat slow, so this can speed up repeated calls to the function
    to test different thresholds.
    :return: bw: The binary image produced by thresholding
             sorted_pix: The sorted list of pixels from which the threshold was calculated
    """
    # Sort pixels in image, if a sorted array is not already available
    if sorted_pix is None:
        if bw_mask is not None:
            sorted_pix = np.sort(im[bw_mask])
        else:
            sorted_pix = np.sort(im.flatten())

    # Convert to a proportion if we appear to have got a percentage
    if proportion > 1:
        proportion /= 100
        print('The threshold exceeds 1; it will be divided by 100.')

    # Invert PERCENT if DARK
    if dark:
        proportion = 1 - proportion

    # Get threshold
    threshold, sorted_pix = percentage_threshold(sorted_pix, proportion, True)

    # Threshold to get darkest or lightest objects
    if dark:
        bw = im <= threshold
    else:
        bw = im > threshold

    # Apply mask
    if bw_mask is not None:
        bw = bw & bw_mask

    return bw, sorted_pix

def get_image_with_object_sizes(bw, labeled, num_objects):
    object_sizes = ndimage.sum(bw, labels=labeled, index=range(1, num_objects+1))
    # Create a new image where each pixel is labeled with the size of the object it belongs to
    size_image = np.zeros_like(labeled, dtype=np.int32)
    for i in range(1, num_objects+1):
        size_image[labeled == i] = object_sizes[i-1]
    return size_image

def clean_segmented_image(bw, min_object_size, min_hole_size):
    """
    Clean a binary image by removing small objects and filling small holes.

    Input:
        bw (numpy.ndarray): the binary image
        min_object_size (int): the minimum size of an object to be kept (in pixels)
        min_hole_size (int): the minimum size of a 'hole' (a region surrounded by detected pixels); smaller holes will be filled in.

    Output:
        bw_clean (numpy.ndarray): the modified binary image
        obj_holes (dict): a dictionary containing the size and location of objects and holes, which can be used to speed up later calls to this function (assuming bw is unchanged).
    """
    # Remove small objects, if necessary
    if min_object_size > 0:
        labeled, num_objects = ndimage.label(bw)
        object_sizes = ndimage.sum(bw, labels=labeled, index=range(1, num_objects+1))
        size_image = get_image_with_object_sizes(bw, labeled, num_objects)
        object_mask = size_image >= min_object_size
        bw_clean = object_mask
    else:
        bw_clean = bw
    # Fill in holes, if necessary
    if min_hole_size > 0:
        clean = ~bw_clean
        labeled, num_holes = ndimage.label(clean)
        hole_sizes = ndimage.sum(clean, labels=labeled, index=range(1, num_holes+1))
        hole_sizes = get_image_with_object_sizes(clean, labeled, num_holes)
        hole_mask = hole_sizes < min_hole_size
        bw_clean = hole_mask
    else:
        bw_clean = bw_clean

    obj_holes = {'object_sizes': object_sizes, 'hole_sizes': hole_sizes}

    return bw_clean, obj_holes

def seg_iuwt(vessel_data, args=None):
    """
    Segment an image using the Isotropic Undecimated Wavelet Transform (IUWT, or 'a trous' transform).
    
    Required VESSEL_DATA properties: IM
    Optional VESSEL_DATA properties: BW_MASK
    Set VESSEL_DATA properties:      BW
    
    ARGS contents:
      IUWT_DARK - TRUE if vessels are darker than their surroundings (e.g.
      fundus images), FALSE if they are brighter (e.g. fluorescein
      angiograms; default FALSE).
      IUWT_INPAINTING - TRUE if pixels outside the FOV should be replaced
      with the closest pixel values inside before computing the IUWT.  This
      reduces boundary artifacts.  It is more useful with fluorescein
      angiograms, since the artifacts here tend to produce bright features
      that are more easily mistaken for vessels (default FALSE).
      IUWT_W_LEVELS - a numeric vector containing the wavelet levels that
      should (default 2-3).
      IUWT_W_THRESH - threshold defined as a proportion of the pixels in the
      image or FOV (default 0.2, which will detect ~20% of the pixels).
      IUWT_PX_REMOVE - the minimum size an object needs to exceed in order to
      be kept, defined as a proportion of the image or FOV (default 0.05).
      IUWT_PX_FILL - the minimum size of a 'hole' (i.e. an undetected region
      entirely surrounded by detected pixels), defined as a proportion of the
      image or FOV.  Smaller holes will be filled in (default 0.05).
    
    Returns:
      args - Updated arguments
      cancelled - Flag indicating if segmentation was cancelled
    """
    
    # Set up default input arguments in case none are available
    args = {
        'iuwt_dark': True,
        'iuwt_inpainting': False,
        'iuwt_w_levels': [2, 3],
        'iuwt_w_thresh': 0.2,
        'iuwt_px_remove': 0.05,
        'iuwt_px_fill': 0.05
    }

    #dist_inpainting
    scale = numel(vessel_data.bw) / 100
    w = iuwt_vessels(vessel_data.im, args["iuwt_w_levels"])

    bw, _ = percentage_segment(w, args["iuwt_w_thresh"], args["iuwt_dark"], vessel_data.bw_mask)

    vessel_data.bw = bw
    bw_clean, _ = clean_segmented_image(vessel_data.bw, args["iuwt_px_remove"] * scale, args["iuwt_px_fill"] * scale)
    vessel_data.bw = bw_clean

    vessel_data.dark_vessels = args["iuwt_dark"]
