from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)

from .helper import Helper


class RevertResampleContourWorker(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _result_image_path, _image_name_path, _data_directory, _temp_directory,
                 _dict_of_images, _apply_to_all, _edit_cnt_convex_hull,
                 _current_editing_tab, parent=None):
        super(RevertResampleContourWorker, self).__init__(parent)

        self.result_image_path = _result_image_path
        self.image_path = _image_name_path

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.dict_of_images = _dict_of_images
        self.apply_to_all = _apply_to_all
        self.edit_cnt_convex_hull = _edit_cnt_convex_hull
        self.current_editing_tab = _current_editing_tab

    @pyqtSlot()
    def work(self):
        if self.apply_to_all and len(self.dict_of_images) > 0:
            for image_name in self.dict_of_images.values():
                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)

                Helper.remove_resampled_contour_file_from_temp(
                    self.temp_directory, result_image_path)

        else:

            Helper.remove_resampled_contour_file_from_temp(
                self.temp_directory, self.result_image_path)

        self.sig_done.emit(1, 'done')
