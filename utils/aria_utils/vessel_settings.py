import os

class Vessel_Settings:
    """
    VESSEL_SETTINGS A range of display settings for use with VESSEL_DATA.
    These settings are concerned with, for example, whether to paint selections
    or centre lines, or to use calibrated values or pixels for measurements.
    """
    def __init__(self):
        self.calibrate = False
        self.calibration_value = 1
        self.calibration_unit = 'px'
        self.show_centre_line = True
        self.show_diameters = True
        self.show_edges = True
        self.show_highlighted = True
        self.show_labels = True
        self.show_orig = False
        self.show_optic_disc = True
        self.plot_lines = False
        self.plot_markers = True
        self.show_spacing = 1
        self.col_centre_line = [0, 0, 1]
        self.col_diameters = [1, 0, 0] # Included diameters
        self.col_diameters_ex = [0, 0, 1] # Excluded diameters
        self.col_edges = [1, 0, 0]
        self.col_highlighted = [1, 1, 0]
        self.col_labels = [0, 0, 1]
        self.col_optic_disc = [0, .85, 0]
        self.double_buffer = True
        self.last_path = os.getcwd()
        self.prompt = False