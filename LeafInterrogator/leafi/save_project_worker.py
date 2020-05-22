from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)

from .helper import Helper


class SaveProjectWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _data_directory, _temp_directory, _project_file_path,
                 _compression_type, parent=None):
        super(SaveProjectWorkerThread, self).__init__(parent)

        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.project_file_path = _project_file_path
        self.compression_type = _compression_type
        self.list_of_images = []

    @pyqtSlot()
    def work(self):
        compression_type = self.compression_type.split(".")[-1].split(")")[0].strip()

        Helper.make_tarfile(self.project_file_path, self.temp_directory, compression_type)

        self.sig_done.emit(1, 'done')
