from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)

from .detect_contours import DetectContour
from .helper import Helper
from .metadata import Metadata


class ResampleContourWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _number_of_points, _result_image_path, _image_name_path, _data_directory, _temp_directory,
                 _dict_of_images, _apply_to_all, _edit_cnt_convex_hull, _flip_all_to_same_side,
                 _current_editing_tab, parent=None):
        super(ResampleContourWorkerThread, self).__init__(parent)

        self.number_of_points = _number_of_points
        self.result_image_path = _result_image_path
        self.image_path = _image_name_path

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.dict_of_images = _dict_of_images
        self.apply_to_all = _apply_to_all
        self.edit_cnt_convex_hull = _edit_cnt_convex_hull
        self.flip_all_to_same_side = _flip_all_to_same_side
        self.current_editing_tab = _current_editing_tab

    @pyqtSlot()
    def work(self):
        if self.apply_to_all and len(self.dict_of_images) > 0:
            for image_name in self.dict_of_images.values():
                image_path = Helper.get_image_path_from_image_name(
                    self.temp_directory, image_name, self.data_directory)
                dc = DetectContour(image_path)
                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)
                if self.edit_cnt_convex_hull:
                    contour = Helper.load_convexhull_from_csv(self.temp_directory,
                                                              result_image_path)
                else:
                    contour = Helper.load_contours_from_csv(self.temp_directory,
                                                            result_image_path)

                resampled_points = dc.sample_contour(contour,
                                                     self.number_of_points,
                                                     self.temp_directory,
                                                     result_image_path)

                md = Metadata(self.temp_directory, result_image_path)

                if self.flip_all_to_same_side:
                    try:
                        # if the leaflet is left leaflet we do not rotate it
                        if 'left' in md.metadata_dict.get('leaflet_position', ''):
                            landmarks = Helper.get_landmarks_from_meta_data(contour, self.temp_directory,
                                                                            result_image_path)
                            # landmarks[[0, 1]] = landmarks[[1, 0]]
                            md.update_metadata({'landmarks': landmarks})
                            md.save_metadata()

                            resampled_points = dc.sample_contour(contour,
                                                                 self.number_of_points,
                                                                 self.temp_directory,
                                                                 result_image_path, landmarks)

                            resampled_points = [resampled_points[0][::-1]]

                            Helper.save_resampled_csv(self.temp_directory,
                                                      resampled_points,
                                                      result_image_path)
                            continue
                        elif not 'left' in md.metadata_dict.get('leaflet_position', '') or \
                                not 'terminal' in md.metadata_dict.get('leaflet_position', ''):
                            landmarks = Helper.get_landmarks_from_meta_data(contour, self.temp_directory,
                                                                            result_image_path)
                            resampled_points, landmarks = Helper.flip_contour_points(
                                resampled_points, landmarks)

                            md.update_metadata({'landmarks': landmarks})
                            md.save_metadata()

                    except TypeError as e:
                        print(e)
                        pass

                        # resampled_points = Helper.flip_contour_points(
                        #     resampled_points)

                Helper.save_resampled_csv(self.temp_directory,
                                          resampled_points, result_image_path)
        else:
            try:
                dc = DetectContour(self.image_path)
            except IndexError:
                self.sig_done.emit(0, 'Cannot detect contour!')
                return

            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, self.image_path)

            if self.edit_cnt_convex_hull:
                contour = Helper.load_convexhull_from_csv(self.temp_directory,
                                                          result_image_path)
            else:
                contour = Helper.load_contours_from_csv(self.temp_directory,
                                                        result_image_path)

            resampled_points = dc.sample_contour(contour,
                                                 self.number_of_points,
                                                 self.temp_directory,
                                                 self.result_image_path)

            # meta_data = Helper.read_metadata_from_csv(self.temp_directory,
            #                                           result_image_path)
            # if self.flip_all_to_same_side: #and self.current_editing_tab == 1:
            #     resampled_points = Helper.flip_contour_points(
            #         resampled_points)
            #     if 'left' in meta_data.get('leaflet_position', ''):
            #         resampled_points = [resampled_points[0][::-1]]
            #
            #     elif not 'left' in meta_data.get('leaflet_position', '') or \
            #             not 'terminal' in meta_data.get('leaflet_position', ''):
            #         landmarks = Helper.get_landmarks_from_meta_data(contour, self.temp_directory,
            #                                                         result_image_path)
            #         landmarks[[0, 1]] = landmarks[[1, 0]]
            #         meta_data['landmarks'] = landmarks
            #
            #         Helper.save_metadata_to_csv(self.temp_directory, result_image_path, meta_data)
            #
            #         resampled_points = Helper.flip_contour_points(
            #             resampled_points)

            Helper.save_resampled_csv(self.temp_directory, resampled_points,
                                      self.result_image_path)

        self.sig_done.emit(1, 'done')
