from utils.aria_utils.vessel_settings import Vessel_Settings
import matplotlib.pyplot as plt
import numpy as np

class Vessel_Data:
    def __init__(self, settings=None):
        self._settings = settings if settings is not None else Vessel_Settings()
        self.im_orig = None
        self.im = None
        self.bw_mask = None
        self.bw = None
        self.dark_vessels = True
        self.file_name = None
        self.file_path = None
        self.args = {}
        self.vessel_list = []
        self.optic_disc_centre = None
        self.optic_disc_diameter = None
        self.optic_disc_mask = None
        self.val_selected_vessel_ind = -1
        self.id_val = 0
        self.thin = None
    
    def delete_vessels(self, inds=None):
        if inds is None:
            self.vessel_list = []
        else:
            for index in sorted(inds, reverse=True):
                del self.vessel_list[index]
        # Update the image if displayed
        #self.update_image_lines([], True)
    
    def update_image_lines(obj, h=None, repaint_all=False):
        # Search for figure with the right name for this image
        if h is None or not plt.fignum_exists(h):
            h = obj.get_figure()
            # If there is no figure, don't do anything
            if h is None:
                return

        # Make figure active
        plt.figure(h.number)

        # Identify the lines and labels already present on the image
        lines = [line for line in plt.gca().get_lines()]
        labels = [label for label in plt.gca().get_texts()]
        rects = [rect for rect in plt.gca().patches if isinstance(rect, plt.Rectangle)]

        # Don't repaint all by default
        if repaint_all:
            # Delete previous lines and labels if there
            for line in lines:
                line.remove()
            for label in labels:
                label.remove()
            for rect in rects:
                rect.remove()
            lines, labels, rects = [], [], []

        # Check whether any vessels at all
        if obj.num_vessels == 0:
            return

        # Only paint new lines if none already present
        if not lines:
            # Paint optic disc, if it's available
            if obj.optic_disc_centre is not None and obj.optic_disc_diameter is not None:
                # Show optic disc as circle... which is a rectangle with very, very
                # rounded corners
                disc_rc = obj.optic_disc_centre
                diam = obj.optic_disc_diameter
                plt.gca().add_patch(
                    plt.Rectangle(
                        (disc_rc[2] - diam / 2, disc_rc[1] - diam / 2),
                        diam,
                        diam,
                        edgecolor=obj.settings.col_optic_disc,
                        facecolor='none',
                        visible=False,
                        tag='optic_disc',
                        linewidth=1,
                    )
                )

                # If inner and outer mask regions are set, show them too
                if obj.optic_disc_mask is not None:
                    min_dist = (0.5 + obj.optic_disc_mask[0]) * diam
                    max_dist = (0.5 + obj.optic_disc_mask[1]) * diam

                    # Show inner and outer regions
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (disc_rc[2] - min_dist, disc_rc[1] - min_dist),
                            min_dist * 2,
                            min_dist * 2,
                            edgecolor=obj.settings.col_optic_disc,
                            facecolor='none',
                            visible=False,
                            tag='optic_disc',
                            linewidth=1,
                        )
                    )
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (disc_rc[2] - max_dist, disc_rc[1] - max_dist),
                            max_dist * 2,
                            max_dist * 2,
                            edgecolor=obj.settings.col_optic_disc,
                            facecolor='none',
                            visible=False,
                            tag='optic_disc',
                            linewidth=1,
                        )
                    )

            # Somewhat convoluted but improves painting speed by rather
            # a lot (perhaps 5-10 times), and also improves toggling
            # visible / invisible speed.
            # Because centre lines and vessel edges will either all be
            # shown or none at all, each can be plotted as a single
            # 'line' object rather than separate objects for each
            # vessel.  To do so, they need to be converted into single
            # vectors, with NaN values where points should not

    def add_vessels(obj, v):
        for ii in range(len(v)):
            v[ii].vessel_data = obj
        if not obj.vessel_list:
            obj.vessel_list = v
        else:
            obj.vessel_list.extend(v)
    
    def show(self):
        plt.imshow(self.im, cmap='gray')
        #cent = np.vstack([vessel.centre.T for vessel in self.vessel_list])
        print(f"Total vessels to show: {len(self.vessel_list)}")
        for v in self.vessel_list:
            temp = v.centre
            plt.plot(temp[1, :], temp[0, :], 'b', linewidth=1)
        plt.show()
        return
