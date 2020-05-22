from PyQt5.QtCore import pyqtSignal

from .helper import Helper
from .worker_thread_interface import WorkerThread


class PerformProcessWorkerThread(WorkerThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _controller_obj, _data_directory, _temp_directory, _apply_to_all,
                 _dict_of_images, load_selected, _process_name, parent=None):
        super(PerformProcessWorkerThread, self).__init__(parent)

        self.controller_obj = _controller_obj
        self.data_directory = _data_directory
        self.temp_directory = _temp_directory

        self.apply_to_all = _apply_to_all
        self.load_selected = load_selected
        self.process_name = _process_name
        self.dict_of_images = _dict_of_images

        self.data_float = []
        self.results = []
        self.ls = None

    def work(self):

        # if the re-sampled contour is exist we use it, else we
        # will get the original contour
        if self.apply_to_all:
            if self.load_selected.need_resampling:
                try:
                    cnts = self.controller_obj.get_all_resampled_contours()
                except:
                    self.sig_done.emit(0, 'Error on get all resampled contours!')
                    return
                if not cnts:
                    self.sig_done.emit(0, 'Error on get all resampled contours!')
                    return
            else:
                self.controller_obj.create_all_contours_dict_list()
                cnts = self.controller_obj.contours_dict_list

            for contour in cnts:
                try:
                    if self.load_selected.need_resampling:
                        # ls = self.load_selected({'contour': contour['resampled_contour']})
                        self.load_selected._contour = contour['resampled_contour']
                    else:
                        # ls = self.load_selected({'contour': contour['contour']})
                        self.load_selected._contour = contour['contour']
                except Exception as e:
                    print(e)
                    continue

                result = self.load_selected.run()

                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, contour['image_name'])
                class_name, class_value = Helper.get_class_from_metadata(
                    self.temp_directory, result_image_path)

                temp = {'color_rgb': contour['color_rgb'],
                        'image_name': contour['image_name'],
                        'image_counter':
                            [k for k, value in self.dict_of_images.items()
                             if value in contour['image_name']][0],
                        'process_name': self.process_name,
                        'class_value': class_value
                        }
                result.update(temp)
                self.results.append(result)

        else:
            image_path = self.controller_obj.get_current_image_path()
            if image_path is None:
                self.sig_done.emit(0, 'Unable to find current image!')
                return
            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, image_path)

            if self.load_selected.need_resampling:
                contour = Helper.load_resampled_from_csv(self.temp_directory,
                                                         result_image_path)
                if not contour:
                    self.sig_done.emit(0, "Can not perform the operation!\n"
                                          "Please make sure you resampled the "
                                          "contours before doing this process!")
                    return
            else:
                contour = Helper.load_contours_from_csv(
                    self.temp_directory, result_image_path)

            image_name, ext = Helper.separate_file_name_and_extension(
                image_path)

            color = [1.0, 0.0, 0.0]
            self.controller_obj.image_name_cnt_color_map_list[image_name + ext] = color

            self.results = []
            if self.load_selected.need_resampling:
                ls = self.load_selected(contour)
            else:
                ls = self.load_selected(contour)

            result = ls.run()

            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, image_name + ext)
            class_name, class_value = Helper.get_class_from_metadata(
                self.temp_directory, result_image_path)

            temp = {'color_rgb': color,
                    'image_name': image_name + ext,
                    'image_counter':
                        [k for k, value in self.dict_of_images.items()
                         if value in image_name + ext][0],
                    'process_name': self.process_name,
                    'class_value': class_value
                    }
            result.update(temp)
            self.results.append(result)

        if not self.results:
            self.sig_done.emit(0, 'done')
            return

        self.ls = self.load_selected
        self.sig_done.emit(1, 'done')
