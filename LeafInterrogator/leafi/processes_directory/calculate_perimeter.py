import cv2
from ..process import Process


class Perimeter(Process):
    def __init__(self, **kwargs):
        self._contour = kwargs.get('contour')
        self._perimeter = None

        self._process_name = "Perimeter"
        self._parent_name = "Basic Measures"
        self._need_resampling = False

        self._plot_type = 'bar_plot'


    def run(self):
        self._perimeter = cv2.arcLength(self._contour[0], True)

        return {self.process_name: round(self._perimeter, 5)}
