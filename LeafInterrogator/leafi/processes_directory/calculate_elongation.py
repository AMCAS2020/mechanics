import cv2
from ..process import Process


class Elongation(Process):
    """
    Elongation, is an other concept based on eccentricity (ratio of the
    Width and Length of the minimum bounding box), thus:
    Elongation = 1 − W/L

    Elongation takes values between 0 and 1. The Elongation value of the a
    symmetric shape in all axis like square or circle is 0.
    """
    def __init__(self, **kwargs):
        self._contour = kwargs.get('contour')
        self._elongation = None

        self._process_name = "Elongation"
        self._parent_name = "Basic Measures"
        self._need_resampling = False

        self._plot_type = 'bar_plot'


    def run(self):
        for cnt in self._contour:
            # print("minAreaRect=", cv2.minAreaRect(cnt))

            center, width_height, rotate_angle = cv2.minAreaRect(cnt)
            width, height = sorted(width_height)
            # print("(width / height)=", 1 - (width / height))
            self._elongation = round(1 - (width / height), 5)

        # --------- base on moments
#            m = cv2.moments(cnt)
#            x = m['mu20'] + m['mu02']
#            y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
#            self._elongation = (x + y ** 0.5) / (x - y ** 0.5)

        return {self.process_name: round(self._elongation, 5)}
