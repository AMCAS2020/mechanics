from PyQt5.QtCore import (pyqtSignal, QThread)

from .helper import Helper
from .metadata import Metadata


class DeleteFeatureFromMetadataWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _controller_obj, _temp_directory, _apply_to_all,
                 _user_reply, parent=None):
        super(DeleteFeatureFromMetadataWorkerThread, self).__init__(parent)

        self.controller_obj = _controller_obj
        self.temp_directory = _temp_directory
        self._apply_to_all = _apply_to_all
        self.user_reply = _user_reply

    def work(self):
        selected = self.controller_obj.metadata_table_widget.currentRow()
        if self._apply_to_all:
            feature_name = self.controller_obj.metadata_table_widget.item(selected, 1).text()
            if self.user_reply:
                for image_name in self.controller_obj.dict_of_images.values():
                    result_image_path = Helper.get_result_image_path_from_image_name(
                        self.controller_obj.temp_directory, image_name)
                    # remove from table
                    self.controller_obj.metadata_table_widget.removeRow(selected)
                    # remove from the metadata file

                    md = Metadata(self.temp_directory, result_image_path)
                    try:
                        del md.metadata_dict[feature_name]
                    except:
                        pass
                    md.save_metadata()

        else:
            selected = self.controller_obj.metadata_table_widget.currentRow()

            feature_name = self.controller_obj.metadata_table_widget.item(selected, 1).text()
            if self.user_reply:
                # remove from table
                self.controller_obj.metadata_table_widget.removeRow(selected)
                # remove from the metadata file
                result_image_path = self.controller_obj.get_current_result_image_path()

                md = Metadata(self.temp_directory, result_image_path)
                del md.metadata_dict[feature_name]
                md.save_metadata()

        self.sig_done.emit(1, 'done')
