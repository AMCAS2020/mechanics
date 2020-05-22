from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from .helper import Helper
from .metadata import Metadata
from .worker_thread_interface import WorkerThread


class SaveUserInputMetadataWorkerThread(WorkerThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _controller_obj, _data_directory, _temp_directory,
                 _dict_of_images, _user_metadata_dict, _apply_to_all_empty_fields,
                 _apply_to_all_with_same_class, _apply_to_all_with_same_parent, _apply_to_all, parent=None):
        super(SaveUserInputMetadataWorkerThread, self).__init__(parent)

        self.controller_obj = _controller_obj
        self.data_directory = _data_directory
        self.temp_directory = _temp_directory

        self.dict_of_images = _dict_of_images
        self.user_metadata_dict = _user_metadata_dict
        self.apply_to_all = _apply_to_all
        self.apply_to_all_with_same_class = _apply_to_all_with_same_class
        self.apply_to_all_with_same_parent =_apply_to_all_with_same_parent
        self.apply_to_all_empty_fields = _apply_to_all_empty_fields

    def work(self):
        result_image_path = self.controller_obj.get_current_result_image_path()
        old_md = Metadata(self.temp_directory, result_image_path)

        as_class_param = None
        for i in range(self.controller_obj.metadata_table_rows):
            param = self.controller_obj.metadata_table_widget.item(i, 1).text()
            if param in old_md.metadata_dict.keys():
                value = ''
            elif self.controller_obj.metadata_table_widget.item(i, 1) is None or \
                            self.controller_obj.metadata_table_widget.item(i, 1) is '':
                # self.controller_obj.warning_message("Can not find value for "
                #                                     "row {} and column {}.".format(i, 0))
                continue
            elif self.controller_obj.metadata_table_widget.item(i, 2) is None or \
                            self.controller_obj.metadata_table_widget.item(i, 2) is '':
                # self.controller_obj.warning_message("Can not find value for "
                #                                     "row {} and column {}.".format(i, 1))
                continue
            if self.controller_obj.metadata_table_widget.cellWidget(i, 0).findChild(
                    type(QtWidgets.QRadioButton())).isChecked():
                as_class_param = self.controller_obj.metadata_table_widget.item(
                    i, 1).text()

                self.user_metadata_dict['as_class_param'] = as_class_param

            value = self.controller_obj.metadata_table_widget.item(i, 2).text()
            self.user_metadata_dict[param] = value
        set1 = set(old_md.metadata_dict.items())
        set2 = set(self.user_metadata_dict.items())
        result_dict = dict(set2 - set1)
        if self.apply_to_all_with_same_class:
            for image_name in self.dict_of_images.values():
                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)

                md = Metadata(self.temp_directory, result_image_path)
                if old_md.get_class_name() == md.get_class_name():
                    # Change the Class
                    md.update_metadata(result_dict)
                    # md.update_metadata({'as_class_param': as_class_param})
                    # But keep the parameter value (if exists)
                    # md.add_metadata({param: value})

                    md.save_metadata()

        elif self.apply_to_all_with_same_parent: #TODO
            for image_name in self.dict_of_images.values():
                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)

                md = Metadata(self.temp_directory, result_image_path)
                print(md.result_image_path())
                if old_md.result_image_path() == md.result_image_path():
                    # Change the Class
                    md.update_metadata(result_dict)
                    # But keep the parameter value (if exists)
                    # md.add_metadata({param: value})

                    md.save_metadata()


        elif self.apply_to_all_empty_fields: # TODO currently works only for tables with empty class
            for image_name in self.dict_of_images.values():
                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)

                md = Metadata(self.temp_directory, result_image_path)
                if not md.get_class_name():
                    # Change the Class
                    md.update_metadata(result_dict)
                    # But keep the parameter value (if exists)
                    # md.add_metadata({param: value})

                    md.save_metadata()

        elif self.apply_to_all:
            for image_name in self.dict_of_images.values():
                result_image_path = Helper.get_result_image_path_from_image_name(
                    self.temp_directory, image_name)

                md = Metadata(self.temp_directory, result_image_path)
                # Change the Class
                md.update_metadata(result_dict)
                # But keep the parameter value (if exists)
                # Apply to all
                # md.update_metadata({param: value})

                md.save_metadata()

        else:
            result_image_path = self.controller_obj.get_current_result_image_path()

            md = Metadata(self.temp_directory, result_image_path)
            # Change the Class
            md.update_metadata(result_dict)
            # if not apply to all the class parameter will change
            # md.update_metadata({param: value})

            md.save_metadata()

        if not self.user_metadata_dict:
            self.sig_done.emit(0, '')
            return

        # if np.array(self.controller_obj.edit_contour_opengl_widget.mapped_landmarks).any():
        #     landmarks = \
        #         self.controller_obj.edit_contour_opengl_widget.landmarks_map_back_to_image()
        #
        #     self.user_metadata_dict['landmarks'] = landmarks
        #
        # current_result_image_path = self.controller_obj.get_current_result_image_path()
        #
        # old_dict = Helper.read_metadata_from_csv(self.temp_directory,
        #                                          current_result_image_path)
        # if old_dict:
        #     old_dict.update(self.user_metadata_dict)
        #     self.user_metadata_dict = {}
        #
        #     Helper.save_metadata_to_csv(self.temp_directory,
        #                                 current_result_image_path,
        #                                 old_dict)

        self.sig_done.emit(1, 'done')

    def question_message(self, message, window_title="?"):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText(message)
        msg.setWindowTitle(window_title)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes |
                               QtWidgets.QMessageBox.No)
        msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
        # msg.buttonClicked.connect(
        #     self.leaf_number_question_message_handler)

        reply = msg.exec_()
        if reply == QtWidgets.QMessageBox.Yes:
            return True
        else:
            return False
