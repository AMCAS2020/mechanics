from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QObject)


class ProgressBarWorkerThread(QObject):
    sig_done = pyqtSignal(int)

    # sig_step = pyqtSignal(int, str)  # worker id, step
    # sig_done = pyqtSignal(int)  # worker id
    # sig_msg = pyqtSignal(str)  # message to be shown to user

    def __init__(self, id: int):
        super().__init__()
        self.__id = id

    @pyqtSlot()
    def work(self):
        # while self.running:
        self.sig_done.emit(int(0))

    def work_done(self):
        self.sig_done.emit(int(1))
