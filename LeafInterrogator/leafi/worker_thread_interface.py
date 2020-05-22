from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)


class WorkerThread(QThread):
    """
        The interface for creating worker on another thread
    """
    sig_done = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super(WorkerThread, self).__init__(parent)

    @pyqtSlot()
    def work(self):
        """
            override this method in order to do send the
            operation to another thread
        :return: -
        """
        pass
