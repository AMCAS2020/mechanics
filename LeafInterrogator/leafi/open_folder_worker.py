import shutil

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)

from .helper import Helper


class OpenFolderWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _data_directory, _temp_directory, _folder_path,
                 parent=None):
        super(OpenFolderWorkerThread, self).__init__(parent)

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.folder_path = _folder_path

        self.list_of_images = []

    @pyqtSlot()
    def work(self):
        self.list_of_images = Helper.get_list_of_images(
            self.folder_path)

        for image in self.list_of_images:
            org_path = Helper.build_path(self.folder_path, image)
            new_path = Helper.build_path(self.data_directory, image)
            shutil.copy2(org_path, new_path)

        self.sig_done.emit(1, 'done')
