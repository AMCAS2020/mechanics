from PyQt5.QtCore import (pyqtSignal, QThread)

from .helper import Helper


class SaveAllWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _data_directory, _temp_directory, _destination_folder,
                 _image_base_name, _image_prefix, _image_suffix,
                 _selected_image_format, parent=None):
        super(SaveAllWorkerThread, self).__init__(parent)

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.destination_folder = _destination_folder
        self.image_base_name = _image_base_name
        self.image_prefix = _image_prefix
        self.image_suffix = _image_suffix
        self.selected_image_format = _selected_image_format

    def work(self):
        Helper.save_process_results(self.temp_directory, self.destination_folder)

        Helper.rename_image_and_csv_files(self.temp_directory,
                                          self.destination_folder,
                                          self.image_base_name,
                                          self.image_prefix,
                                          self.image_suffix,
                                          self.selected_image_format)

        self.sig_done.emit(1, 'done')
