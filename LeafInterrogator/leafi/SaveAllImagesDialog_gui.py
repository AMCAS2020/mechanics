# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'save_all_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class Ui_SaveAllDialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(320, 200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(320, 200))
        Dialog.setMaximumSize(QtCore.QSize(320, 200))
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setMaximumSize(QtCore.QSize(16777215, 100))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.suffix_lbl = QtWidgets.QLabel(self.widget)
        self.suffix_lbl.setObjectName("suffix_lbl")
        self.gridLayout.addWidget(self.suffix_lbl, 2, 0, 1, 1)
        self._images_prefix = QtWidgets.QLineEdit(self.widget)
        self._images_prefix.setObjectName("_images_prefix")
        self.gridLayout.addWidget(self._images_prefix, 1, 1, 1, 1)
        self._images_name = QtWidgets.QLineEdit(self.widget)
        self._images_name.setObjectName("_images_name")
        self.gridLayout.addWidget(self._images_name, 0, 1, 1, 1)
        self._images_suffix = QtWidgets.QLineEdit(self.widget)
        self._images_suffix.setObjectName("_images_suffix")
        self.gridLayout.addWidget(self._images_suffix, 2, 1, 1, 1)
        self.name_lbl = QtWidgets.QLabel(self.widget)
        self.name_lbl.setObjectName("name_lbl")
        self.gridLayout.addWidget(self.name_lbl, 0, 0, 1, 1)
        self.prefix_lbl = QtWidgets.QLabel(self.widget)
        self.prefix_lbl.setObjectName("prefix_lbl")
        self.gridLayout.addWidget(self.prefix_lbl, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.widget)
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setItemText(6, "")
        self.verticalLayout.addWidget(self.comboBox)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self._reject_button_clicked = False

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(lambda: self.reject(Dialog))
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self._images_name, self._images_prefix)
        Dialog.setTabOrder(self._images_prefix, self._images_suffix)
        Dialog.setTabOrder(self._images_suffix, self.comboBox)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Images Information"))
        self.suffix_lbl.setText(_translate("Dialog", "Suffix:"))
        self.name_lbl.setText(_translate("Dialog", "Name:"))
        self.prefix_lbl.setText(_translate("Dialog", "Prefix:"))
        self.comboBox.setCurrentText(_translate("Dialog", "Default (keep format of the original images)"))
        self.comboBox.setItemText(0, _translate("Dialog", "Default (keep format of the original images)"))
        self.comboBox.setItemText(1, _translate("Dialog", "Portable Network Graphics (.png)"))
        self.comboBox.setItemText(2, _translate("Dialog", "Joint Photographic Experts Group (.jpg)"))
        self.comboBox.setItemText(3, _translate("Dialog", "bitmap (.bmp)"))
        self.comboBox.setItemText(4, _translate("Dialog", "Tagged Image File Format (.tiff)"))
        self.comboBox.setItemText(5, _translate("Dialog", "Joint Photographic Experts Group (.jpeg)"))

    def reject(self, dialog):
        self._reject_button_clicked = True
        dialog.reject()
