import numpy as np
from scipy.sparse import find, triu
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label, regionprops
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes, distance_transform_edt, grey_dilation, grey_erosion, binary_opening, binary_closing
from scipy.interpolate import *
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
from scipy.interpolate import RectBivariateSpline
from utils.aria_utils.vessel import Vessel
import matplotlib.pyplot as plt
from utils.aria_utils.matlab_bwmorph import thin, spur
from utils.aria_utils.matlab_imreconstruct import imreconstruct
from skimage.filters import rank
import math
from scipy.interpolate import RectBivariateSpline, CubicSpline


def binary_to_thinned_segments(bw, spur_length=10, clear_branches_dist=True):
    #bw_thin = binary_erosion(bw, iterations=1)
    bw_thin = thin(bw, math.inf)

    neighbour_count = rank.sum(bw_thin.astype(np.uint8), np.ones((3,3)))
    bw_branches = np.logical_and((neighbour_count > 3), bw_thin) 
    bw_ends = np.logical_and((neighbour_count <= 2), bw_thin)
    
    bw_segments = np.logical_and(bw_thin, np.logical_not(bw_branches))

    bw_terminal = imreconstruct(bw_ends.astype('uint8'), bw_segments.astype('uint8'))

    bw_thin[bw_terminal & ~binary_dilation(bw_terminal, iterations=spur_length)] = False
    
    #bw_thin = spur(bw_thin) # TODO: Add spur again

    bw_thin = thin(bw_thin, math.inf)

    neighbour_count = rank.sum(bw_thin.astype(np.uint8), np.ones((3,3)))
    bw_branches = np.logical_and((neighbour_count > 3), bw_thin) 
    bw_segments = np.logical_and(bw_thin, np.logical_not(bw_branches))
    
    if clear_branches_dist:
        dist = distance_transform_edt(~bw)
        bw[bw_branches] = False
        dist2 = distance_transform_edt(~bw)
        bw_segments = bw_thin & (dist == dist2)
        return bw_segments, bw_thin, bw_branches, dist
    else:
        return bw_segments, bw_thin, bw_branches

# histc: The function counts the number of elements of X that fall in the histogram bins defined by bins.
# This is a Python implementation of the MATLAB histc function.
# https://github.com/jernejvivod/languages-similarity-analysis/blob/a40b9455569da2a868a4242f419da29d26d6a450/lib_naloga2/triplet_extractor.py
def histc(x, bins):
    map_to_bins = np.digitize(x, bins) 	# Get indices of the bins to which each value in input array belongs.
    res = np.zeros(bins.shape)
    for el in map_to_bins:
        res[el-1] += 1 					# Increment appropriate bin.
    return res

def thinned_vessel_segments(bw, spur_length=10, min_px=2, remove_extreme=False, clear_branches_dist=True):
    
    if spur_length is None:
        spur_length = 10
    if min_px is None:
        min_px = 2
    
    bw_segments, bw_thin, _, dist_trans = binary_to_thinned_segments(bw, spur_length, clear_branches_dist)

    labeled_segments = label(bw_segments)
    vessels = [Vessel() for _ in range(labeled_segments.max())]
    remove_inds = np.zeros(len(vessels), dtype=bool)
    
    for ii in range(labeled_segments.max()):
        px_inds = labeled_segments == (ii+1)

        if px_inds.sum() < min_px or (remove_extreme and (dist_trans[px_inds].max() * 2 > px_inds.sum())):
            remove_inds[ii] = True
            continue
        
        row, col = np.where(px_inds)
        
        if len(row) <= 2:
            vessels[ii].centre = [row, col]
            continue
        
        dist = (np.abs(row[:, None] - row) <= 1) & (np.abs(col[:, None] - col) <= 1)

        if np.all(np.diag(dist, 1)):
            vessels[ii].centre = [row, col]
            continue
        
        link = np.argwhere(np.triu(dist, k=1))
        # Get the number of occurrences of each index
        locs = np.arange(0, link.max() + 1)
        n = histc(link.ravel(order='F'), locs)
        #n = np.histogram(link, bins=locs)[0]

        # Find indices that only occur once (i.e. the end points)
        loc_inds = n == 1
        if np.count_nonzero(loc_inds) != 2:
            remove_inds[ii] = True
            continue

        # Determine the first and last values of the locs vector
        locs[[0, -1]] = locs[loc_inds]
        
        for jj in range(1, len(locs)-1):
            rem_row = np.any(link == locs[jj-1], axis=1)
            rem_links = link[rem_row]
            locs[jj] = rem_links[(rem_links != locs[jj-1])]
            link = link[~rem_row]
        
        vessels[ii].centre = [row[locs-1], col[locs-1]]
    
    if np.any(remove_inds):
        vessels = [v for i, v in enumerate(vessels) if not remove_inds[i]]
    
    if remove_extreme:
        dist_trans = np.where(bw_segments, dist_trans, 0)
    
    if clear_branches_dist:
        dist_trans = distance_transform_edt(np.logical_not(bw_segments))
    
    if remove_extreme or clear_branches_dist:
        if (len(regionprops(labeled_segments))):
            inds_all = np.concatenate([r.coords for r in regionprops(labeled_segments)])
            if inds_all.size == 0:
                dist_max = 0
            else:
                #dist_max = np.max(dist_trans[inds_all])
                dist_max = 0
        else:
            dist_max = 0

    return vessels, dist_max, bw_thin


def spline_centreline(vessels, piece_spacing, remove_invalid=True, spine_fitting=True):

    remove_inds = np.zeros(len(vessels), dtype=bool)

    for ii, v in enumerate(vessels):
        c_points = np.squeeze(np.array([v.centre])).T
        if len(c_points) == 0 or c_points.shape[0] < 2 :
            remove_inds[ii] = True
            continue

        y = c_points
        
        pixel_length = np.concatenate([[0], np.sqrt(np.sum(np.diff(c_points, axis=0)**2, axis=1))])

        x = np.cumsum(np.sqrt(pixel_length))

        n_pieces = max(1, int(np.round(np.sum(pixel_length) / piece_spacing)))
        x_eval = np.arange(np.floor(np.max(x))+1)

        if n_pieces == 1 or not spine_fitting:
            p1 = np.polyfit(x, y[:, 0], 2)
            p2 = np.polyfit(x, y[:, 1], 2)
            cent = np.stack([np.polyval(p1, x_eval), np.polyval(p2, x_eval)], axis=0)
            pd1 = np.polyder(p1)
            pd2 = np.polyder(p2)
            der = np.stack([np.polyval(pd1, x_eval), np.polyval(pd2, x_eval)], axis=0)
        else:
            # Determine breakpoints
            b = np.linspace(x[0], x[-1], n_pieces)
            e = np.eye(len(b))

            # compute the spline
            spl = CubicSpline(b, e)(x)
            tempvar = np.linalg.lstsq(spl, y, rcond=-1)[0]

            # compute the piecewise polynomial
            pp = CubicSpline(b, tempvar)

            # evaluate the piecewise polynomial
            cent = pp(x_eval).T

            """
            print("--------------------")
            print(tempvar.shape)
            print(y.shape)
            pprint(tempvar)

            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(b, tempvar, 'o', label='data')
            ax.plot(x_eval, cent.T, label="S")
            plt.show()

            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(b, e, 'o', label='data')
            ax.plot(x, spl, label="S")
            ax.set_title(f"Total points: {b.shape}")
            plt.show()
            """

            # compute the derivative
            pd = pp.derivative()
            #pd.c *= np.array([3, 2, 1])[:, None]
            der = pd(x_eval).T

        vessels[ii].centre = cent

        #Convert derivative values to unit tangents at each point
        normals = np.dot([[0, 1], [-1, 0]], der)
        normals = normalize_vectors(normals)
        vessels[ii].angles = normals.T

    if remove_invalid and np.any(remove_inds):
        vessels_filtered = []
        for removable, vessel in zip(remove_inds, vessels):
            if (not removable):
                vessels_filtered.append(vessel)
        vessels = vessels_filtered

    return vessels


def normalize_vectors(v):
    return v / np.linalg.norm(v, axis=0, keepdims=True)

def make_image_profiles(vessels, im, width, method='linear', bw_mask=None):
    if method is None or method == '':
        method = 'linear'

    if bw_mask is None:
        bw_mask = np.ones_like(im, dtype=bool)
    elif bw_mask.shape != im.shape or bw_mask.dtype != bool:
        print('MAKE_IMAGE_PROFILES:MASK_SIZE', 'BW_MASK must be a logical array the same size as IM; no mask will be applied.')
        bw_mask = np.ones_like(im, dtype=bool)

    if im.size == 0 or im.ndim != 2:
        return vessels
    
    centre = np.vstack([np.stack(v.centre, axis=0).T for v in vessels])
    angles = np.vstack([v.angles for v in vessels]).astype(np.float32)

    inc = np.arange(width) - (width-1)/2
    im_profiles_rows = centre[:, 0, np.newaxis] + angles[:, 0, np.newaxis] * inc[np.newaxis, :]
    im_profiles_cols = centre[:, 1, np.newaxis] + angles[:, 1, np.newaxis] * inc[np.newaxis, :]

    if np.issubdtype(im.dtype, np.integer):
        im = im.astype(np.float32)

    im[~bw_mask] = np.nan

    # Create a function object for 2D interpolation
    interp_func = RectBivariateSpline(range(im.shape[0]), range(im.shape[1]), im, kx=1, ky=1)

    # Call the interpolation function at the specified points
    all_profiles = interp_func.ev(im_profiles_rows.ravel(), im_profiles_cols.ravel())

    # Reshape the output array to match the dimensions of the input array
    all_profiles = all_profiles.reshape(im_profiles_cols.shape)

    current_ind = 0
    for ii, v in enumerate(vessels):
        n_rows = v.centre.shape[0]
        rows = slice(current_ind, current_ind + n_rows)
        v.im_profiles = all_profiles[rows]
        v.im_profiles_rows = im_profiles_rows[rows]
        v.im_profiles_cols = im_profiles_cols[rows]
        current_ind += n_rows

    return vessels

def centre_spline_fit(vessel_data, args):
    args_default = {'spline_piece_spacing': 10, 'centre_spurs': 10, 'centre_min_px': 10,
                    'centre_remove_extreme': True, 'centre_clear_branches_dist_transform': True}
    args.update(args_default)
    cancelled = False
    bw = vessel_data.bw
    im = vessel_data.im
    if bw.size == 0 or im.size == 0 or bw.shape != im.shape:
        raise ValueError('CENTRE_SPLINE_FIT requires binary and original images of the same size (i.e. BW and IM properties in VESSEL_DATA)')

    # Get thinned centreline segments
    vessels, dist_max, bw_thin = thinned_vessel_segments(bw, args['centre_spurs'], args['centre_min_px'], 
                                                args_default['centre_remove_extreme'], args['centre_clear_branches_dist_transform'])

    vessel_data.thin = bw_thin

    if (not args["only_thinning"]):

        # Refine the centreline and compute angles by spline-fitting
        vessels = spline_centreline(vessels, args['spline_piece_spacing'], True)

        # Compute image profiles, using the distance transform of the segmented
        # image to ensure the profile length will be long enough to contain the
        # vessel edges
        # If there is no vessel diameter estimate already, could use this (though
        # it tends to over-estimate)
        # d = distance_transform_edt(~bw)
        # width = int(np.ceil(np.max(d) * 4))
        width = int(np.ceil(dist_max * 4))
        if width % 2 == 0:
            width += 1
        # Make the image profiles - not using a mask here, since this plays havoc
        # with later filtering because the filter smears out NaNs towards the
        # vessel and can prevent the detection of perfectly good edges
        make_image_profiles(vessels, im, width, 'linear')

        # Make sure the list in VESSEL_DATA is empty, then add the vessels
        vessel_data.delete_vessels()
        vessel_data.add_vessels(vessels)
    
    return args, cancelled
