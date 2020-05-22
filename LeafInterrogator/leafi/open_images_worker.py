import shutil

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)

from .helper import Helper


class OpenImagesWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _data_directory, _temp_directory, _file_paths,
                 parent=None):
        super(OpenImagesWorkerThread, self).__init__(parent)

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.file_paths = _file_paths

        self.list_of_images = []

    @pyqtSlot()
    def work(self):
        for file in self.file_paths:
            shutil.copy2(file, self.data_directory)

        self.list_of_images = Helper.get_list_of_images(
            self.data_directory)

        self.sig_done.emit(1, "done")
