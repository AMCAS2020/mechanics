# -*- coding: utf-8 -*-

from .SaveAllImagesDialog_gui import Ui_SaveAllDialog


class ExtensionSaveAllImageDialog(Ui_SaveAllDialog):
    @property
    def selected_value_combobox(self):
        return str(self.comboBox.currentText())

    @property
    def selected_index_combobox(self):
        return self.comboBox.currentIndex()

    @property
    def all_images_base_name(self):
        return self._images_name

    @property
    def all_images_prefix(self):
        return self._images_prefix

    @property
    def all_images_suffix(self):
        return self._images_suffix

    @property
    def reject_button_clicked(self):
        return self._reject_button_clicked
