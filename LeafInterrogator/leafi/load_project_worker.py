from PyQt5.QtCore import pyqtSignal

from .helper import Helper
from .worker_thread_interface import WorkerThread


class LoadProjectWorkerThread(WorkerThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _controller_cls, _data_directory, _temp_directory, _project_file_path,
                 parent=None):
        super(LoadProjectWorkerThread, self).__init__(parent)

        self.controller_cls = _controller_cls
        self.data_directory = _data_directory
        self.temp_directory = _temp_directory
        self.project_file_path = _project_file_path

        self.list_of_images = []

    def work(self):
        compression_type = self.project_file_path.split(".")[-1].strip()
        Helper.extract_tarfile(self.project_file_path, self.temp_directory, compression_type)

        self.sig_done.emit(1, 'done')
