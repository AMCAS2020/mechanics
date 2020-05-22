import logging
import os

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)

from .detect_contours import DetectContour
from .helper import Helper
from .metadata import Metadata


class CropImagesWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _data_directory, _temp_directory, _image_path, _result_image_path,
                 _result_crop_dir, _threshold, _minimum_perimeter, _maximum_perimeter,
                 _approx_factor, _contour_thickness, _convexhull_thickness,
                 _approximation_checkbox, _remove_squares_checkbox, parent=None):
        super(CropImagesWorkerThread, self).__init__(parent)
        self.logger = logging.getLogger(__name__)

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.list_of_images = []

        self.image_path = _image_path
        self.result_image_path = _result_image_path
        self.result_crop_dir = _result_crop_dir
        self.threshold = _threshold
        self.minimum_perimeter = _minimum_perimeter
        self.maximum_perimeter = _maximum_perimeter
        self.approx_factor = _approx_factor
        self.contour_thickness = _contour_thickness
        self.convexhull_thickness = _convexhull_thickness
        self.approximation_checkbox = _approximation_checkbox
        self.remove_squares_checkbox = _remove_squares_checkbox

    @pyqtSlot()
    def work(self):

        if ("data_dir" in self.image_path) == False:
            self.sig_done.emit(0, 'Image has already been cropped')
            return
        print(self.image_path)
        print("data_dir" in self.image_path)    
        dc = DetectContour(self.image_path)
        contours = dc.find_leaf_contour(
            self.threshold, 255, self.minimum_perimeter,
            self.maximum_perimeter,
            self.remove_squares_checkbox)
        if contours == None or len(contours) == 0:
            self.sig_done.emit(0, 'No contours to crop')
            return
        dc.draw_contour(contours, -1)
        dc.save_image(self.result_image_path)

        # save metadata --------------------------------
        metadata_dict = {
            'threshold': self.threshold,
            'approx_factor': self.approx_factor,
            'minimum_perimeter': self.minimum_perimeter,
            'contour_thickness': self.contour_thickness,
            'convexhull_thickness': self.contour_thickness,
        }
        md = Metadata(self.temp_directory, self.result_image_path, **metadata_dict)
        md.save_metadata()

        if len(contours) > 1:
            Helper.save_contours_csv(self.temp_directory,
                                     contours,
                                     self.result_image_path)

            bounds = dc.find_image_bounding_rect(contours)
            # dc.crop_image(bounds, self.result_crop_dir)
            dc.change_color_of_leaf_main_image(self.image_path, contours, self.result_crop_dir)
            dc.crop_result_image(bounds, self.result_crop_dir)

            # self.update_after_crop(result_crop_dir)

            # find contour for each cropped image again and save them
            # in the "cropped/" folder
            images = Helper.get_list_of_images(self.result_crop_dir)
            #  Sort image names so that order of images matches that of contours
            #images.sort()
            i = 0
            for image_name in images:
                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)

                dc = DetectContour(result_image_path)
                try:
                    contours = dc.find_leaf_contour(
                        self.threshold, 255, self.minimum_perimeter,
                        self.maximum_perimeter,
                        self.remove_squares_checkbox,
                        bordersize=0)
                    if contours is None:
                        # delete the existing files
                        Helper.delete_image_and_files(self.temp_directory,
                                                      image_name)
                        data_dir = Helper.get_image_path_from_image_name(
                            self.temp_directory, image_name, self.data_directory)

                        try:
                            if not os.listdir(os.path.dirname(data_dir)):
                                os.rmdir(os.path.dirname(data_dir))

                            # delete the image from data_dir
                            data_dir = Helper.build_path(
                                self.temp_directory, 'data_dir', image_name)
                            os.remove(data_dir)
                        except FileNotFoundError:
                            self.logger.error('Error on finding file to delete.', exc_info=True)
                            pass

                        self.update_after_change_number_of_images()
                        continue
                except Exception:
                    self.logger.error('Error on cropping image', exc_info=True)
                    continue

                Helper.save_contours_csv(self.temp_directory, contours,
                                         result_image_path)

                # save metadata --------------------------------
                metadata_dict = {
                    'threshold': self.threshold,
                    'approx_factor': self.approx_factor,
                    'minimum_perimeter': self.minimum_perimeter,
                    'contour_thickness': self.contour_thickness,
                    'convexhull_thickness': self.contour_thickness,
                    'bounding_rect': [bounds[i]],
                }

                md = Metadata(self.temp_directory, result_image_path, **metadata_dict)
                md.save_metadata()

                i += 1

        self.sig_done.emit(1, 'done')
