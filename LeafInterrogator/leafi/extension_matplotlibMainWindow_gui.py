# -*- coding: utf-8 -*-

import matplotlib
from PyQt5 import QtCore, QtWidgets

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar

from .matplotlibMainWindow_gui import Ui_matplotlibMainWindow
from .matplotlib_canvas import LocalizedMplPlots


class ExtensionMatplotlibMainWindow(Ui_matplotlibMainWindow):
    def __init__(self, MainWindow):
        self.setupUi(MainWindow)
        MainWindow.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget(MainWindow)

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.main_widget.setFocus()
        MainWindow.setCentralWidget(self.main_widget)

        self.plot_canvas = None

        self.plot_canvas = LocalizedMplPlots(self.main_widget)

        navi_toolbar = NavigationToolbar(self.plot_canvas, self.main_widget)
        self.layout.addWidget(self.plot_canvas)
        self.layout.addWidget(navi_toolbar)

        MainWindow.statusBar().showMessage("Done!", 2000)

    def save_plot(self, path):
        self.plot_canvas.save(path=path)
