import os

from PyQt5.QtCore import pyqtSignal

from .helper import Helper
from .worker_thread_interface import WorkerThread


class UpdateWorkingContourWorkerThread(WorkerThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _controller_obj, _data_directory, _temp_directory,
                 _dict_of_images, parent=None):
        super(UpdateWorkingContourWorkerThread, self).__init__(parent)

        self.controller_obj = _controller_obj
        self.data_directory = _data_directory
        self.temp_directory = _temp_directory

        self.dict_of_images = _dict_of_images

    def work(self):

        if self.dict_of_images is None:
            list_of_images = []
            for image_name in Helper.get_list_of_images(self.controller_obj.data_directory):
                if self.controller_obj.selected_working_contours == 'convex_hull':
                    if self.controller_obj.is_image_cropped(image_name):
                        cropped_image_path = Helper.get_cropped_image_path(
                            self.controller_obj.temp_directory, image_name)
                        cropped_path = os.path.dirname(cropped_image_path)
                        for img in Helper.get_list_of_images(cropped_path):
                            list_of_images.append(img)
                    else:
                        list_of_images.append(image_name)

                if self.controller_obj.selected_working_contours == 'leaf':
                    if self.controller_obj.is_image_cropped(image_name):
                        cropped_image_path = Helper.get_cropped_image_path(
                            self.controller_obj.temp_directory, image_name)
                        cropped_path = os.path.dirname(cropped_image_path)
                        for img in Helper.get_list_of_images(cropped_path):
                            list_of_images.append(img)
                    else:
                        list_of_images.append(image_name)
                else:
                    if self.controller_obj.is_image_split(image_name):
                        if self.controller_obj.is_image_cropped(image_name):
                            cropped_split_path = Helper.get_cropped_split_leaflets_path(
                                self.controller_obj.temp_directory, image_name)
                            if self.controller_obj.selected_working_contours == 'leaflets':
                                for img in Helper.get_list_of_images(cropped_split_path):
                                    list_of_images.append(img)
                            elif self.controller_obj.selected_working_contours == 'lateral':
                                for img in Helper.get_list_of_images(cropped_split_path):
                                    if 'lateral' in img:
                                        list_of_images.append(img)
                            elif self.controller_obj.selected_working_contours == 'terminal':
                                for img in Helper.get_list_of_images(cropped_split_path):
                                    if 'terminal' in img:
                                        list_of_images.append(img)
                        else:
                            split_path = Helper.get_split_leaflets_directory(
                                self.controller_obj.temp_directory, image_name)
                            if self.controller_obj.selected_working_contours == 'leaflets':
                                for img in Helper.get_list_of_images(split_path):
                                    list_of_images.append(img)
                            elif self.controller_obj.selected_working_contours == 'lateral':
                                for img in Helper.get_list_of_images(split_path):
                                    if 'lateral' in img:
                                        list_of_images.append(img)
                            elif self.controller_obj.selected_working_contours == 'terminal':
                                for img in Helper.get_list_of_images(split_path):
                                    if 'terminal' in img:
                                        list_of_images.append(img)
                    else:
                        if self.controller_obj.is_image_cropped(image_name):
                            cropped_image_path = Helper.get_cropped_image_path(
                                self.controller_obj.temp_directory, image_name)
                            cropped_path = os.path.dirname(cropped_image_path)

                            for img_name in Helper.get_list_of_images(cropped_path):
                                if self.controller_obj.is_image_split(img_name):
                                    cropped_split_path = Helper.get_cropped_split_result_leaflets_path(
                                        self.controller_obj.temp_directory, img_name)

                                    if self.controller_obj.selected_working_contours == 'leaflets':
                                        for img in Helper.get_list_of_images(cropped_split_path):
                                            list_of_images.append(img)
                                    elif self.controller_obj.selected_working_contours == 'lateral':
                                        for img in Helper.get_list_of_images(cropped_split_path):
                                            if 'lateral' in img:
                                                list_of_images.append(img)
                                    elif self.controller_obj.selected_working_contours == 'terminal':
                                        for img in Helper.get_list_of_images(cropped_split_path):
                                            if 'terminal' in img:
                                                list_of_images.append(img)

            self.controller_obj.dict_of_images = {}
            counter = 1
            for image_name in list_of_images:
                self.controller_obj.dict_of_images[counter] = image_name
                counter += 1
        else:
            self.controller_obj.dict_of_images = self.dict_of_images

        self.controller_obj.total_images_in_folder = len(self.controller_obj.dict_of_images)

        self.controller_obj.current_image_number = 1

        self.controller_obj.set_image_numbers_label(
            self.controller_obj.current_image_number,
            self.controller_obj.total_images_in_folder)

        self.sig_done.emit(1, 'done')
