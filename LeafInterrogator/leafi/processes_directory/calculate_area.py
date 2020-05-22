import cv2
from ..process import Process


class Area(Process):
    def __init__(self, **kwargs):
        self._contour = kwargs.get('contour')
        self._area = None

        self._process_name = "Area"
        self._parent_name = "Basic Measures"
        self._need_resampling = False

        self._plot_type = 'bar_plot'

    def run(self):

        self._area = cv2.contourArea(self._contour[0], True)

        return {self.process_name: round(self._area, 5)}


