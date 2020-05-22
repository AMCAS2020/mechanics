# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exportshapespacedialog.ui'
#
# Created by: PyQt5 UI code generator 5.7.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 150)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self._shape_space_plot_output_format_lbl = QtWidgets.QLabel(Dialog)
        self._shape_space_plot_output_format_lbl.setObjectName("_shape_space_plot_output_format_lbl")
        self.horizontalLayout.addWidget(self._shape_space_plot_output_format_lbl)
        self._shape_space_plot_output_format_combo_box = QtWidgets.QComboBox(Dialog)
        self._shape_space_plot_output_format_combo_box.setMaximumSize(QtCore.QSize(100, 16777215))
        self._shape_space_plot_output_format_combo_box.setObjectName("_shape_space_plot_output_format_combo_box")
        self._shape_space_plot_output_format_combo_box.addItem("")
        self._shape_space_plot_output_format_combo_box.addItem("")
        self._shape_space_plot_output_format_combo_box.addItem("")
        self.horizontalLayout.addWidget(self._shape_space_plot_output_format_combo_box)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self._shape_space_plot_output_format_lbl.setText(_translate("Dialog", "Shape space plot output format:"))
        self._shape_space_plot_output_format_combo_box.setItemText(0, _translate("Dialog", "svg"))
        self._shape_space_plot_output_format_combo_box.setItemText(1, _translate("Dialog", "pdf"))
        self._shape_space_plot_output_format_combo_box.setItemText(2, _translate("Dialog", "png"))
