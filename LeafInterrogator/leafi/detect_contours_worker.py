import traceback

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)

from .detect_contours import DetectContour
from .helper import Helper
from .metadata import Metadata


class DetectContoursWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _data_directory, _temp_directory, _data_directory_name,
                 _image_path, _result_image_path, _dict_of_images,
                 _threshold, _minimum_perimeter, _maximum_perimeter,
                 _approx_factor, _contour_thickness, _convexhull_thickness,
                 _remove_squares_checkbox, _approximation_checkbox,
                 _apply_to_all_images_checkbox, _draw_contour_checkbox,
                 _draw_convexhull_checkbox, parent=None):
        super(DetectContoursWorkerThread, self).__init__(parent)

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.list_of_images = []
        self.dict_of_images = _dict_of_images
        self.data_directory_name = _data_directory_name

        self.image_path = _image_path
        self.result_image_path = _result_image_path
        self.threshold = _threshold
        self.minimum_perimeter = _minimum_perimeter
        self.maximum_perimeter = _maximum_perimeter
        self.approx_factor = _approx_factor
        self.contour_thickness = _contour_thickness
        self.convexhull_thickness = _convexhull_thickness
        self.approximation_checkbox = _approximation_checkbox
        self.remove_squares_checkbox = _remove_squares_checkbox
        self.apply_to_all_images_checkbox = _apply_to_all_images_checkbox
        self.draw_contour_checkbox = _draw_contour_checkbox
        self.draw_convexhull_checkbox = _draw_convexhull_checkbox
        self.not_founded_contours_counter = 0
        self.list_of_images_which_contours_not_founded = []
        self.cnt_points = []

    @pyqtSlot()
    def work(self):
        contours = []
        # apply to all images
        if self.apply_to_all_images_checkbox:
            for key, image_name in self.dict_of_images.items():
                if ("leaflet" in image_name) or ("terminal" in image_name):
                    continue
                image_path = Helper.get_image_path_from_image_name(
                    self.temp_directory, image_name, self.data_directory_name)

                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)
                bounding_rect = ''
                try:
                    dc = DetectContour(image_path)
                    bounding_rect = Helper.get_bounding_rect_from_meta_data(
                        self.temp_directory, result_image_path)

                    contours = dc.find_leaf_contour(
                        self.threshold, 255, self.minimum_perimeter,
                        self.maximum_perimeter,
                        self.remove_squares_checkbox, bounding_rect)

                    if not contours:
                        # add boarder to the image (it helps if shape touch
                        # the border of the image )
                        dc.add_border_to_scaled_image()
                        contours = dc.find_leaf_contour(
                            self.threshold, 255, self.minimum_perimeter,
                            self.maximum_perimeter,
                            self.remove_squares_checkbox, bounding_rect)
                        # remove the added boarder
                        dc.remove_border_from_scaled_image()

                    if not contours:
                        if image_name not in self.list_of_images_which_contours_not_founded:
                            self.list_of_images_which_contours_not_founded.append(
                                image_name)

                        self.not_founded_contours_counter += 1

                    elif image_name in self.list_of_images_which_contours_not_founded:
                        a = self.list_of_images_which_contours_not_founded

                        self.list_of_images_which_contours_not_founded = \
                            [x for x in a if x != image_name]

                except Exception as e:
                    if image_name not in self.list_of_images_which_contours_not_founded:
                        self.list_of_images_which_contours_not_founded.append(
                            image_name)
                    self.not_founded_contours_counter += 1

                # apply contour to all images
                if self.draw_contour_checkbox:
                    if self.approximation_checkbox:
                        try:
                            approx_contours = dc.contour_points_approx(contours,
                                                                       self.approx_factor)
                            contours = approx_contours
                            dc.draw_contour(approx_contours, self.contour_thickness)
                        except:
                            Helper.critical_message(
                                "Unable to find or draw approximate contour.")
                    else:
                        try:
                            dc.draw_contour(contours, self.contour_thickness)
                        except:
                            Helper.critical_message("Unable to draw contour.")

                # apply convex hull to all images
                if self.draw_convexhull_checkbox:
                    try:
                        convex_hull = dc.find_convex_hull(contours)
                        Helper.save_convexhull_to_csv(self.temp_directory, [convex_hull], result_image_path)
                        dc.draw_convexhull(contours, self.convexhull_thickness)
                    except:
                        Helper.critical_message("Unable to draw convexhull.")

                dc.save_image(result_image_path)
                # save contour for all images
                Helper.save_contours_csv(self.temp_directory, contours, result_image_path)

                # save metadata ------------------------
                metadata_dict = {
                    'threshold': self.threshold,
                    'approx_factor': self.approx_factor,
                    'minimum_perimeter': self.minimum_perimeter,
                    'contour_thickness': self.contour_thickness,
                    'convexhull_thickness': self.contour_thickness,
                    'bounding_rect': bounding_rect,
                }

                md = Metadata(self.temp_directory, result_image_path, **metadata_dict)
                md.save_metadata()

            # load only the current image
            # self.load_result_image(self.result_image_path)

            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, self.image_path)

            self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                            result_image_path)
        else:
            bounding_rect = ''
            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, self.image_path)

            current_img_name, _ = Helper.separate_file_name_and_extension(
                self.image_path,
                keep_extension=True)

            try:
                dc = DetectContour(self.image_path)
                bounding_rect = Helper.get_bounding_rect_from_meta_data(
                    self.temp_directory, result_image_path)
                contours = dc.find_leaf_contour(
                    self.threshold, 255, self.minimum_perimeter,
                    self.maximum_perimeter,
                    self.remove_squares_checkbox, bounding_rect,bordersize=0)
                if contours == None:
                    contours = []
                if (contours == None) or (not contours):
                    # add boarder to the image (it helps if shape touch
                    # the border of the image )
                    dc.add_border_to_scaled_image()
                    contours = dc.find_leaf_contour(
                        self.threshold, 255, self.minimum_perimeter,
                        self.maximum_perimeter,
                        self.remove_squares_checkbox, bounding_rect)

                    # remove the added boarder
                    dc.remove_border_from_scaled_image()
                if not contours:
                    if current_img_name not in self.list_of_images_which_contours_not_founded:
                        self.list_of_images_which_contours_not_founded.append(
                            current_img_name)
                    self.not_founded_contours_counter += 1
                    # self.warning_message(
                    #     'No contour found for "{}"'.format(current_img_name))

                elif current_img_name in \
                        self.list_of_images_which_contours_not_founded:
                    a = self.list_of_images_which_contours_not_founded

                    self.list_of_images_which_contours_not_founded = \
                        [x for x in a if x != current_img_name]
            except Exception as e:
                if current_img_name not in self.list_of_images_which_contours_not_founded:
                    self.list_of_images_which_contours_not_founded.append(
                        current_img_name)
                self.not_founded_contours_counter += 1
                # self.critical_message("Unable to detect the contour.\n" + str(
                #     type(e)))

            # draw contour of the current image
            if self.draw_contour_checkbox:
                # print("len(contours)=",len(contours), self.approximation_checkbox, self.contour_thickness)
                if self.approximation_checkbox:
                    try:
                        approx_contours = dc.contour_points_approx(contours,
                                                                   self.approx_factor)
                        self.cnt_points = approx_contours
                        contours = approx_contours
                        dc.draw_contour(approx_contours, self.contour_thickness)
                    except:
                        pass
                        # Helper.critical_message(
                        #     "Unable to find or draw approximate contour.")
                else:
                    if contours != None and len(contours) > 0:
                        _interior = "_cropped_" in result_image_path.split('/')[-1]
                        dc.draw_contour(contours, self.contour_thickness,interior=_interior)

            # draw convex hull of the current image
            if self.draw_convexhull_checkbox:
                try:
                    convex_hull = dc.find_convex_hull(contours)
                    Helper.save_convexhull_to_csv(self.temp_directory, [convex_hull], result_image_path)
                    dc.draw_convexhull(contours, self.convexhull_thickness)
                except:
                    tb = traceback.format_exc()
                    print(tb)
                    self.sig_done.emit(0, "Unable to draw convexhull.")
                    return

                self.cnt_points = contours

            dc.save_image(result_image_path)

            # save metadata -------------------------------------------
            metadata_dict = {
                'threshold': self.threshold,
                'approx_factor': self.approx_factor,
                'minimum_perimeter': self.minimum_perimeter,
                'contour_thickness': self.contour_thickness,
                'convexhull_thickness': self.contour_thickness,
                'bounding_rect': bounding_rect,
            }
            md = Metadata(self.temp_directory, result_image_path, **metadata_dict)

            md.save_metadata()

            Helper.save_contours_csv(self.temp_directory, contours,
                                     result_image_path)

        self.sig_done.emit(1, 'done')
