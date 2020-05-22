from .exportshapespacedialog_gui import Ui_Dialog


class ExtensionExportShapeSpaceDialog(Ui_Dialog):
    @property
    def shape_space_plot_output_format_combo_box(self):
        return self._shape_space_plot_output_format_combo_box
