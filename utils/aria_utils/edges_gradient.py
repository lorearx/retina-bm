import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage

def create_vessel_profile_masks(vessels, bw, bw_mask=None):
    if bw_mask is None:
        bw_mask = np.array([])
    elif bw_mask.size > 0 and bw_mask.shape != bw.shape:
        print('Warning: BW and BW_MASK must be the same size - BW_MASK will be ignored')
        bw_mask = np.array([])

    rows = np.round(np.concatenate([v.im_profiles_rows for v in vessels]))
    cols = np.round(np.concatenate([v.im_profiles_cols for v in vessels]))
    inds_coords = (cols - 1) * bw.shape[0] + rows
    valid_coords = (rows >= 1) & (cols >= 1) & (rows <= bw.shape[0]) & (cols <= bw.shape[1])
    if bw_mask.size > 0:
        valid_coords[valid_coords] = bw_mask.flat[inds_coords[valid_coords].astype(int)]
    bw_vessel_profiles_all = np.zeros(inds_coords.shape, dtype=bool)
    bw_vessel_profiles_all[valid_coords] = bw.flat[inds_coords[valid_coords].astype(int)]

    bw_vessel_profiles = get_centreline_object(bw_vessel_profiles_all)

    print(bw_vessel_profiles.shape)

    bw_region_profiles = bw_vessel_profiles == bw_vessel_profiles_all
    bw_region_profiles[~valid_coords] = False

    n_profiles = np.array([len(v.centre) // 2 for v in vessels])
    mask_regions = np.split(bw_region_profiles, np.cumsum(n_profiles))
    vessel_regions = np.split(bw_vessel_profiles, np.cumsum(n_profiles))

    return vessel_regions, mask_regions

def get_centreline_object(bw):
    c = bw.shape[1] // 2
    col_centre = bw[:, c]
    for ii in range(c+1, bw.shape[1]):
        col_centre = col_centre & bw[:, ii]
        if not np.any(col_centre):
            bw[:, ii:] = False
            break
        bw[:, ii] = col_centre
    col_centre = bw[:, c]
    for ii in range(c-1, -1, -1):
        col_centre = col_centre & bw[:, ii]
        if not np.any(col_centre):
            bw[:, :ii+1] = False
            break
        bw[:, ii] = col_centre
    return bw

def sub2ind2d(sz, rowSub, colSub):
    linIndex = (colSub - 1) * sz[0] + rowSub
    return linIndex

def find_most_connected_crossings(crossings, column, region_length):
    if np.isscalar(column):
        bw_region = np.abs(crossings - column) <= region_length
    else:
        bw_region = np.abs(np.subtract.outer(crossings, column)) <= region_length
    crossings[~ndimage.morphology.binary_propagation(bw_region, mask=np.isfinite(crossings))] = np.nan
    finite_crossings = np.isfinite(crossings)
    if np.sum(finite_crossings, axis=1) <= 1:
        return
    lab, num = ndimage.label(finite_crossings)
    if num == 1:
        crossings[lab != 1] = np.nan
    else:
        n_labs = np.histogram(lab, bins=num, range=(1, num))[0]
        lab[lab > 0] = n_labs[lab[lab > 0] - 1]
    bw_region = lab == np.max(lab, axis=1)[:, np.newaxis]
    crossings[~bw_region] = np.nan


def find_closest_crossing(crossings, column, search_previous):
    search_next = ~search_previous
    cross = np.full((crossings.shape[0],), np.nan)
    if np.any(search_previous):
        cross[search_previous] = find_previous_crossing(crossings[search_previous, :], column)
    if np.any(search_next):
        cross[search_next] = find_next_crossing(crossings[search_next, :], column)
    return cross

def find_next_crossing(crossings, column):
    crossings[crossings < column] = np.nan
    cross_next = np.nanmin(crossings, axis=1)
    return cross_next

def find_previous_crossing(crossings, column):
    crossings[crossings > column] = np.nan
    cross_prev = np.nanmax(crossings, axis=1)
    return cross_prev

def compute_discrete_2d_derivative(prof):
    prof_2d = prof[:, :-1] + prof[:, 1:] - 2 * prof
    return prof_2d

def find_maximum_gradient_columns(prof, region_length):
    c = prof.shape[1] // 2
    region_length = int(np.ceil(region_length)) + 1
    if region_length >= c:
        region_length = c
    prof[:, 1:-1] = prof[:, 2:] - prof[:, :-2]
    prof[:, :c-region_length] = np.nan
    prof[:, c+region_length:] = np.nan
    left_col = np.nanargmax(prof[:, :c], axis=1)
    right_col = np.nanargmin(prof[:, c:], axis=1) + c
    return left_col, right_col

def get_side(im_profiles_rows, im_profiles_cols, rows, cols):
    cols_floor = np.floor(cols)
    cols_diff = cols - np.floor(cols)
    inds_floor = sub2ind2d(im_profiles_rows.shape, rows, cols_floor)
    inds_floor_plus = inds_floor + im_profiles_rows.shape[0]
    side_rows = im_profiles_rows.flat[inds_floor] * (1 - cols_diff) + im_profiles_rows.flat[inds_floor_plus] * cols_diff
    side_cols = im_profiles_cols.flat[inds_floor] * (1 - cols_diff) + im_profiles_cols.flat[inds_floor_plus] * cols_diff
    side = np.column_stack((side_rows, side_cols))
    return side

def set_edges_2nd_derivative(vessels, bw, bw_mask, smooth_scale_parallel=1, smooth_scale_perpendicular=0.1, enforce_connectedness=True):
    if smooth_scale_parallel is None:
        smooth_scale_parallel = 1
    if smooth_scale_perpendicular is None:
        smooth_scale_perpendicular = 0.1
    
    vessel_regions, mask_regions = create_vessel_profile_masks(vessels, bw, bw_mask)
    
    for ii in range(len(vessels)):
        vessels[ii].side1 = np.full(vessels[ii].centre.shape, np.nan)
        vessels[ii].side2 = vessels[ii].side1
        
        im_profiles = vessels[ii].im_profiles
        im_profiles_cols = vessels[ii].im_profiles_cols
        im_profiles_rows = vessels[ii].im_profiles_rows
        n_profiles = im_profiles.shape[0]
        
        dark_vessels = vessels[ii].dark
        if dark_vessels:
            im_profiles = -im_profiles
        
        c = int(np.ceil(im_profiles.shape[1] / 2))
        
        bw_vessel_profiles = vessel_regions[ii]
        binary_sums = np.sum(bw_vessel_profiles, axis=1)
        width_est = np.median(binary_sums[bw_vessel_profiles[:, c-1].astype(bool)])
        if width_est > c - 1:
            width_est = c - 1
        elif not np.isfinite(width_est):
            continue
        
        bw_regions = mask_regions[ii]
        
        im_profiles_closest = np.copy(im_profiles)
        print(im_profiles_closest.shape)
        print(bw_regions.shape)
        im_profiles_closest[~bw_regions] = np.nan
        prof_mean = np.nanmean(im_profiles_closest, axis=0)
        
        l_mean_col, r_mean_col = find_maximum_gradient_columns(prof_mean, width_est)
        width_est = r_mean_col - l_mean_col
        if not np.isfinite(width_est):
            continue
        
        gv = gaussian_filter1d(np.ones(int(np.ceil(np.sqrt(width_est * smooth_scale_parallel)))), np.sqrt(width_est * smooth_scale_parallel))
        gh = gaussian_filter1d(np.ones(int(np.ceil(np.sqrt(width_est * smooth_scale_perpendicular)))), np.sqrt(width_est * smooth_scale_perpendicular))
        im_profiles = np.apply_along_axis(lambda x: np.convolve(gh, np.convolve(gv, x, mode='same'), mode='same'), axis=1, arr=im_profiles)
        
        if dark_vessels:
            vessels[ii].im_profiles = -im_profiles
        else:
            vessels[ii].im_profiles = im_profiles
        
        im_profiles_2d = compute_discrete_2d_derivative(im_profiles)
        im_profiles_2d[~bw_regions] = np.nan
        
        diffs = np.diff(im_profiles_2d, axis=1)
        cross_offsets = -im_profiles_2d[:, :-1] / diffs
        cross_offsets[cross_offsets >= 1] = np.nan
        cross_offsets[cross_offsets < 0] = np.nan
        cross = np.tile(np.arange(im_profiles.shape[1] - 1), (im_profiles.shape[0], 1)) + cross_offsets
        
        cross_rising = np.copy(cross)
        cross_rising[diffs > 0] = np.nan
        cross_rising[cross_rising > c] = np.nan
        cross_falling = np.copy(cross)
        cross_falling[diffs < 0] = np.nan
        cross_falling[cross_falling < c] = np


def edges_max_gradient(vessel_data, args=None):
    args_default = {
        "smooth_parallel": 1,
        "smooth_perpendicular": 0.1,
        "enforce_connectivity": True
    }

    if args and isinstance(args, dict):
        args = {**args_default, **args}
    else:
        args = args_default
    
    vessels = vessel_data.vessel_list
    bw = vessel_data.bw
    bw_mask = vessel_data.bw_mask
    
    set_edges_2nd_derivative(vessels, bw, bw_mask, args["smooth_parallel"], args["smooth_perpendicular"], args["enforce_connectivity"])
    
    vessel_data.clean_vessel_list(1)
    
    return args
