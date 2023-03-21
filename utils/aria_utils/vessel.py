class Vessel:
    """
    A VESSEL object is a container for all data and functions relating to a vessel or vessel segment.
    """
    
    def __init__(self):
        # Important value properties, not modified directly
        self.prop_centre = None         # Centre line points
        self.prop_side1 = None          # Edge points (first side)
        self.prop_side2 = None          # Edge points (second side)
        self.prop_angles = None         # Angles of the vessel, as determined at each centre points (in form of unit vectors giving directions)
        self.prop_keep_inds = None      # Indices of 'good' measurements, i.e. measurements to be output
        
        self.prop_dark = []             # TRUE if vessel is dark, i.e. represented by a 'valley' rather than a 'hill'
        
        # Publically available versions of the above
        self.centre = None              # Centre line points
        self.side1 = None               # Edge points (first side)
        self.side2 = None               # Edge points (second side)
        self.angles = None              # Angles of the vessel, as determined at each centre points
        self.keep_inds = None           # Indices of 'good' measurements, i.e. to be output
        
        self.dark = []                  # TRUE if vessel is dark, i.e. represented by a 'valley' rather than a 'hill'
        
        # Dependent properties, computed from other (publically accessible) properties
        self.diameters = None           # Euclidian distance between side1 and side2, scaled by SCALE_VALUE
        self.offset = None              # Cummulative sum of euclidean distances *between* centre points, scaled by SCALE_VALUE (i.e. offset of a diameter measured along the centre line)
        self.valid = None               # TRUE if this object contains sufficient data for measuring diameters & offsets, and it has both sides and centre properties containing the same numbers of points
        self.string = None              # String representation of the object
        self.num_diameters = None       # Total number of diameters
        self.im_profiles_positive = None # As image profiles, but ensuring that the vessel is 'positive' (i.e. a 'hill' rather than a 'valley')
        self.length_straight_line = None # The straight-line length connecting the vessel's end points (i.e. the first and last 'included' points; those in between are ignored)
        self.length_cumulative = None   # The cumulative sum of the straight-line lengths between all the vessel's centre line points, from the first to the last 'included' point
        self.settings = None            # The VESSEL_SETTINGS object stored in the associated VESSEL_DATA
        
        # Scale properties are obtained from VESSEL_SETTINGS calibration if available
        self.scale_value = None         # The scale value (for multiplication)
        self.scale_unit = None          # The unit (e.g. 'pixels', 'µm')
        
        # General properties
        self.vessel_data = None          # An associated VESSEL_DATA object
        self.im_profiles = None          # Image in which each row is the image profile across corresponding centre points at corresponding angle
        self.im_profiles_rows = None     # (Non-integer) row co-ordinate in original image for each pixel in im_profiles
        self.im_profiles_cols = None     # (Non-integer) column co-ordinates
        self.im_profiles_model = None    # Similar to IM_PROFILES, but diameters may be measured from the profiles after smoothing or fitting a model to the data, in which case the profiles actually used are stored here
        self.highlight_inds = None       # Indices of vessels that should be highlighted on plots