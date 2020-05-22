# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

from .leaf_interrogator_gui import Ui_LeafInterrogatorMainWindow

MINIMAL_VERSION = True

# Detect Contour Tab Default Values
DEFAULT_PETIOLE_WIDTH = 700
DEFAULT_MIN_LEAFLET_LENGTH = 30
DEFAULT_PRUNING_NUM_ITERATIONS = 2
DEFAULT_THRESHOLD = 170
DEFAULT_MIN_PERIMETER = 1300
DEFAULT_MAX_PERIMETER = 8000
DEFAULT_APPROXIMATION_FACTOR = 1
DEFAULT_CONTOUR_LINE_THICKNESS = -1
DEFAULT_CONVEXHULL_LINE_THICKNESS = 1

# Edit Contour Tab Default Values
DEFAULT_NUMBER_OF_SAMPLE_POINTS = 60
DEFAULT_TOTAL_NUMBER_OF_LANDMARKS = 2

# Shape Space Tab Default Values
DEFAULT_NUMBER_OF_HARMONICS = 16
DEFAULT_NUMBER_OF_STD = 1
DEFAULT_NUMBER_OF_STD_CURVES = 1
# Plot
DEFAULT_SHAPE_SPACE_PLOT_POINT_SIZE = 10
DEFAULT_SHAPE_SPACE_PLOT_LIM = 10
DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_NUMBER_OF_ROWS = 3
DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_NUMBER_OF_COLUMNS = 4
DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_IMAGE_ZOOM = 1.0
DEFAULT_SHAPE_SPACE_PLOT_ANNOTATION_IMAGES_ZOOM = 0.08
DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_IMAGE_ROTATION = 0.0

class ExtensionLeafInterrogator(Ui_LeafInterrogatorMainWindow):
    def __init__(self):
        # super(Ui_LeafInterrogatorMainWindow, self).__init__()
        self.statusBar = QtWidgets.QStatusBar(self)
        self.progress_bar = QtWidgets.QProgressBar(self)

        self.shift_landmarks = QtWidgets.QAction(QtGui.QIcon(""), "Shift Landmarks", self)

        # self.position_y_axis = QtWidgets.QAction(QtGui.QIcon(""), "on Y-axis", self)

    @property
    def detect_contours_btn(self):
        return self._find_contour_btn

    @property
    def draw_convexhull_checkbox(self):
        return self._draw_convexhull_checkbox

    @property
    def draw_contour_checkbox(self):
        return self._draw_contour_checkbox

    @property
    def apply_to_all_images_checkbox(self):
        return self._apply_to_all_images_checkbox

    @property
    def threshold(self):
        try:
            return float(self._threshold.text())
        except ValueError:
            return DEFAULT_THRESHOLD

    @property
    def min_perimeter(self):
        try:
            return float(self._min_perimeter.text())
        except ValueError:
            return DEFAULT_MIN_PERIMETER

    @property
    def max_perimeter(self):
        try:
            return float(self._max_perimeter.text())
        except ValueError:
            return DEFAULT_MAX_PERIMETER

    #@property
    #def approx_factor(self):
    #    try:
    #        return float(self._approximation_factor.text())
    #    except ValueError:
    #        return DEFAULT_APPROXIMATION_FACTOR

    @property
    def contour_thickness(self):
        if MINIMAL_VERSION == True:
            return -1
        try:
            return int(self._contour_thickness.text())
        except ValueError:
            return DEFAULT_CONTOUR_LINE_THICKNESS

    @property
    def convexhull_thickness(self):
        try:
            return int(self._convexhull_thickness.text())
        except ValueError:
            return DEFAULT_CONVEXHULL_LINE_THICKNESS

    @property
    def image_opengl_widget(self):
        return self._image_opengl_widget

    @image_opengl_widget.setter
    def image_opengl_widget(self, opengl_widget):
        self._image_opengl_widget = opengl_widget

    @property
    def contour_opengl_widget(self):
        return self._contour_opengl_widget

    @contour_opengl_widget.setter
    def contour_opengl_widget(self, opengl_widget):
        self._contour_opengl_widget = opengl_widget

    @property
    def edit_contour_opengl_widget(self):
        return self._edit_contour_opengl_widget

    @edit_contour_opengl_widget.setter
    def edit_contour_opengl_widget(self, opengl_widget):
        self._edit_contour_opengl_widget = opengl_widget

    @property
    def edit_contour_points_radiobtn(self):
        return self._edit_contour_radiobtn

    @property
    def edit_landmarks_radiobtn(self):
        return self._edit_landmarks_radiobtn

    @property
    def keep_unedited_contour_checkbox(self):
        return self._keep_unedited_cnt_checkbox

    @property
    def show_suggested_landmarks_checkbox(self):
        return self._show_suggested_landmarks_checkbox

    @property
    def show_all_landmarks_checkbox(self):
        return self._show_all_landmarks_checkbox

    @property
    def redefine_landmarks_checkbox(self):
        return self._redefine_landmarks_checkbox

    @property
    def total_number_of_landmarks(self):
        try:
            return self._total_landmarks
        except ValueError:
            return DEFAULT_TOTAL_NUMBER_OF_LANDMARKS

    @property
    def submit_total_landmarks_btn(self):
        return self._submit_total_landmarks_btn

    @property
    def number_of_sample_points(self):
        try:
            return int(self._number_of_points.text())
        except ValueError:
            return DEFAULT_NUMBER_OF_SAMPLE_POINTS

    @property
    def resample_btn(self):
        return self._resample_btn

    @property
    def approximation_checkbox(self):
        return self._approximation_checkbox

    @property
    def edit_cnt_next_btn(self):
        return self._edit_cnt_next_btn

    @property
    def edit_cnt_previous_btn(self):
        return self._edit_cnt_previous_btn

    @property
    def edit_cnt_apply_to_all_checkbox(self):
        return self._edit_cnt_apply_to_all_checkbox

    @property
    def edit_cnt_counter_label(self):
        return self._edit_cnt_counter_lbl

    @property
    def edit_cnt_align_points_to_center_checkbox(self):
        return self._align_points_to_center_checkbox

    @property
    def edit_cnt_project_all_contours_on_screen_checkbox(self):
        return self._show_all_contours_checkbox

    @property
    def registration_tree_widget(self):
        return self._registration_treeWidget

    @property
    def apply_registration_btn(self):
        return self._apply_registration_btn

    @property
    def start_process_btn(self):
        return self._start_process_btn

    @property
    def editing_tabs(self):
        return self._editing_tabs

    @property
    def processes_tree_widget(self):
        return self._processes_treeWidget

    @property
    def result_table_widget(self):
        return self._result_tableWidget

    @property
    def process_parameters_table_widget(self):
        return self._process_parameters_tablewidget

    @property
    def crop_images_btn(self):
        return self._crop_images_btn

    @property
    def apply_process_to_all_contours_checkbox(self):
        return self._apply_process_to_all_contours_checkbox

    @property
    def metadata_table_widget(self):
        return self._metadata_tableWidget

    @property
    def insert_metadata_btn(self):
        return self._insert_metadata_button

    @property
    def add_row_metadata_table_btn(self):
        return self._add_row_button

    @property
    def remove_row_metadata_table_btn(self):
        return self._remove_row_button

    @property
    def remove_squares_checkbox(self):
        return self._remove_squares_checkBox

    @property
    def method_combo_box(self):
        return self._method_combo_box

    @property
    def x_axis_combobox(self):
        return self._x_axis_combo_box

    @property
    def y_axis_combobox(self):
        return self._y_axis_combo_box

    @property
    def z_axis_combobox(self):
        return self._z_axis_combo_box

    @property
    def apply_plot_conf_btn(self):
        return self._apply_shape_space_plot_conf_button

    @property
    def x_axis_percentage_label(self):
        return self._x_percent_lbl

    @x_axis_percentage_label.setter
    def x_axis_percentage_label(self, value):
        self._x_percent_lbl.setText("{:.3f} %".format(value * 100))

    @property
    def y_axis_percentage_label(self):
        return self._y_percent_lbl

    @y_axis_percentage_label.setter
    def y_axis_percentage_label(self, value):
        self._y_percent_lbl.setText("{:.3f} %".format(value * 100))

    @property
    def number_of_harmonics_lbl(self):
        return self._number_of_harmonics_lbl

    @property
    def number_of_harmonics(self):
        try:
            return int(self._number_of_harmonics.text())
        except ValueError:
            return DEFAULT_NUMBER_OF_HARMONICS

    @property
    def flip_all_to_same_side_checkbox(self):
        return self._flip_all_to_same_side_checkbox

    @property
    def total_variance_lbl(self):
        return self._total_variance_lbl

    @total_variance_lbl.setter
    def total_variance_lbl(self, value):
        try:
            self._total_variance_lbl.setText("{:.5f}".format(value))
        except ValueError:
            self._total_variance_lbl.setText("")

    @property
    def show_centroid_check_box(self):
        return self._show_centroid_check_box

    @property
    def show_convex_hull_check_box(self):
        return self._show_convex_hull_check_box

    # @property
    # def tab_colors(self):
    #     return self._tab_colors
    #
    # @property
    # def tab_colors_grid_layout(self):
    #     return self._tab_colors_grid_layout

    @property
    def edited_contour_save_changes_btn(self):
        return self._edited_contour_save_changes_btn

    @property
    def plot_lim(self):
        try:
            return float(self._shape_space_plot_lim.text())
        except ValueError:
            return DEFAULT_SHAPE_SPACE_PLOT_LIM

    @property
    def apply_plot_options_btn(self):
        return self._apply_plot_options_btn

    @property
    def min_leaflet_length(self):
        try:
            return float(self._min_leaflet_length.text())
        except ValueError:
            return DEFAULT_MIN_LEAFLET_LENGTH

    @property
    def petiole_width(self):
        try:
            return float(self._petiole_width.text())
        except ValueError:
            return DEFAULT_PETIOLE_WIDTH

    @property
    def find_leaflets_btn(self):
        return self._find_leaflets_btn

    @property
    def pruning_num_iterations(self):
        try:
            return int(self._pruning_num_iterations.text())
        except ValueError:
            return DEFAULT_PRUNING_NUM_ITERATIONS

    @property
    def split_leaflets_btn(self):
        return self._split_leaflets_btn

    @property
    def contour_leaflet_tab(self):
        return self._contour_leaflet_tab

    @property
    def edit_leaflets_cut_points_check_box(self):
        return self._edit_leaflets_cut_points_checkbox

    @property
    def apply_edited_cut_points_btn(self):
        return self._apply_edited_cut_points_btn

    @property
    def select_cut_points_btn(self):
        return self._select_cut_points_btn

    @property
    def remove_cut_points_btn(self):
        return self._remove_cut_points_btn

    @property
    def add_cut_points_btn(self):
        return self._add_cut_points_btn

    @property
    def resample_whole_leaves_check_box(self):
        return self._resample_whole_leaf_checkbox

    @property
    def resample_lateral_leaflets_check_box(self):
        return self._resample_lateral_leaflet_checkbox

    @property
    def resample_terminal_leaflets_check_box(self):
        return self._resample_terminal_leaflet_checkbox

    @property
    def edit_cnt_convex_hull_checkbox(self):
        return self._edit_cnt_convex_hull_checkbox

    @property
    def shape_space_plot_background_number_of_rows(self):
        try:
            return int(self._shape_space_plot_background_number_of_rows.text())
        except ValueError as e:
            print("Plot number of row entry error! ", e)
            return DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_NUMBER_OF_ROWS

    @property
    def shape_space_plot_background_number_of_columns(self):
        try:
            return int(self._shape_space_plot_background_number_of_columns.text())
        except ValueError as e:
            print("Plot number of columns entry error! ", e)
            return DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_NUMBER_OF_COLUMNS

    @property
    def shape_space_plot_background_image_zoom(self):
        try:
            return float(self._shape_space_plot_background_image_zoom.text())
        except ValueError:
            return DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_IMAGE_ZOOM

    @property
    def shape_space_plot_background_image_rotation(self):
        try: #Convert to radians
            return float(self._shape_space_plot_background_image_rotation.text())/180*3.14159 
        except ValueError:
            return DEFAULT_SHAPE_SPACE_PLOT_BACKGROUND_IMAGE_ROTATION


    @property
    def shape_space_plot_annotation_images_zoom(self):
        try:
            return float(self._shape_space_plot_annotation_images_zoom.text())
        except ValueError as e:
            print(e)
            return DEFAULT_SHAPE_SPACE_PLOT_ANNOTATION_IMAGES_ZOOM

    @property
    def shape_space_plot_show_only_closest_original_contour_image_check_box(self):
        return self._shape_space_plot_show_back_ground_annotation_images_check_box

    @property
    def shape_space_plot_show_only_reconstructed_image_check_box(self):
        return self._shape_space_plot_show_only_reconstructed_image_check_box

    @property
    def a_harmonic_coeff_check_box(self):
        # return self._a_harmonic_coeff_checkbox
        return True

    @property
    def b_harmonic_coeff_check_box(self):
        # return self._b_harmonic_coeff_checkbox
        return True

    @property
    def c_harmonic_coeff_check_box(self):
        # return self._c_harmonic_coeff_checkbox
        return True

    @property
    def d_harmonic_coeff_check_box(self):
        # return self._d_harmonic_coeff_checkbox
        return True

    @property
    def symmetric_components_check_box(self):
        return self._symmetric_components_check_box

    @property
    def asymmetric_components_check_box(self):
        return self._asymmetric_components_check_box

    @property
    def harmonic_coeffs_groupbox(self):
        return self._harmonic_coeffs_groupbox

    @property
    def export_components_btn(self):
        return self._export_components_btn

    @property
    def explore_data_btn(self):
        return self._explore_data_btn

    @property
    def apply_to_all_find_leaflets_check_box(self):
        return self._apply_to_all_find_leaflets_checkBox

    @property
    def apply_to_all_tables_empty_fields_checkbox(self):
        return self._apply_to_all_tables_empty_fields_checkbox

    @property
    def apply_to_all_tables_with_same_class_checkbox(self):
        return self._apply_to_all_tables_with_same_class_checkbox

    @property
    def apply_to_all_tables_with_same_parent(self):
        return self._apply_to_all_tables_with_same_parent


    @property
    def apply_to_all_tables_checkbox(self):
        return self._apply_to_all_tables_checkbox

    @property
    def image_counter(self):
        return self._image_counter_lineEdit

    @image_counter.setter
    def image_counter(self, value):
        self._image_counter_lineEdit.setText(str(value))

    @property
    def edit_cnt_counter(self):
        return self._edit_cnt_counter_lineEdit

    @edit_cnt_counter.setter
    def edit_cnt_counter(self, value):
        self._edit_cnt_counter_lineEdit.setText(str(value))

    @property
    def delete_resampled_contour_btn(self):
        return self._delete_resampled_contour_btn

    @property
    def shape_space_tab_layout(self):
        return self._shape_space_tab_horizontal_layout

    @property
    def automatic_flipping_check_box(self):
        return self._automatic_flipping_checkbox

    @property
    def colors_and_classes_scroll_area(self):
        return self._colors_and_classes_scroll_area

    @property
    def scroll_area_widget_contents_grid_layout(self):
        return self._scroll_area_widget_contents_gridLayout

    @property
    def show_std_ellipse_check_box(self):
        return self._show_std_ellipse_check_box

    @property
    def number_of_std(self):
        try:
            return float(self._number_of_std.text())
        except:
            return DEFAULT_NUMBER_OF_STD

    @property
    def number_of_std_curves(self):
        try:
            return int(self._number_of_std_curves.text())
        except:
            return DEFAULT_NUMBER_OF_STD_CURVES

    @property
    def always_show_closest_image_check_box(self):
        return self._always_show_closest_image_check_box

    @property
    def shape_space_image_graphic_view(self):
        return self._shape_space_image_graphic_view

    @property
    def shape_space_image_lbl(self):
        return self._shape_space_image_lbl

    @property
    def shape_space_plot_point_size(self):
        try:
            return float(self._shape_space_plot_point_size.text())
        except ValueError:
            return DEFAULT_SHAPE_SPACE_PLOT_POINT_SIZE

    @property
    def shape_space_plot_show_back_ground_annotation_images_check_box(self):
        return self._shape_space_plot_show_back_ground_annotation_images_check_box

    @property
    def efd_size_invariant_check_box(self):
        return self._efd_size_invariant_checkbox

    @property
    def efd_normalize_check_box(self):
        return self._efd_normalize_checkbox

    @property
    def show_std_error_ellipse_check_box(self):
        return self._show_std_error_ellipse_check_box

    @property
    def shift_landmarks_anticlockwise_btn(self):
        return self._shift_landmarks_anticlockwise_button

    @property
    def x_axis_processes_radio_btn(self):
        return self._x_axis_processes_radio_button

    @property
    def x_axis_pcs_radio_btn(self):
        return self._x_axis_pcs_radio_button

    @property
    def y_axis_processes_radio_btn(self):
        return self._y_axis_processes_radio_button

    @property
    def y_axis_pcs_radio_btn(self):
        return self._y_axis_pcs_radio_button

    @property
    def z_axis_processes_radio_btn(self):
        return self._z_axis_processes_radio_button

    @property
    def z_axis_pcs_radio_btn(self):
        return self._z_axis_pcs_radio_button

    @property
    def shape_space_components_holder_widget(self):
        return self._shape_space_components_holder_widget

    @property
    def lda_eigen_solver_radio_btn(self):
        return self._lda_eigen_solver_radio_button

    @property
    def lda_svd_solver_radio_btn(self):
        return self._lda_svd_solver_radio_button

    @property
    def lda_svd_shrinkage_check_box(self):
        return self._lda_svd_shrinkage_check_box

    # @property
    # def canvas_widget(self):
    #     return self._canvas_widget
    #
    # @canvas_widget.setter
    # def canvas_widget(self, value):
    #     self._canvas_widget = value

    def create_scene_for_shape_space_image(self):
        self.shape_space_image_scene = QtWidgets.QGraphicsScene()
        self.shape_space_image_graphic_view.setScene(self.shape_space_image_scene)

    def build_processes_tree_widget(self, new_class_dict):
        _translate = QtCore.QCoreApplication.translate
        counter = 0
        self.processes_tree_widget.clear()

        for parent, process_names in new_class_dict.items():
            if not parent:
                item_0 = QtWidgets.QTreeWidgetItem(self.processes_tree_widget)
                self.processes_tree_widget.topLevelItem(counter).setText(0,
                                                                         _translate("LSAMainWindow", process_names))
            else:
                root = QtWidgets.QTreeWidgetItem(self.processes_tree_widget)
                root.isDisabled()
                self.processes_tree_widget.topLevelItem(counter).setText(0,
                                                                         _translate("LSAMainWindow", parent))

                for index, process in enumerate(sorted(process_names)):
                    item = QtWidgets.QTreeWidgetItem(root)
                    self.processes_tree_widget.topLevelItem(counter).child(
                        index).setText(0,
                                       _translate("LSAMainWindow", process))

            counter += 1

    def build_parameter_table_widget(self, parameters_list):
        _translate = QtCore.QCoreApplication.translate
        rows_number = len(parameters_list)
        self.process_parameters_table_widget.setRowCount(rows_number)
        row_counter = 0
        for parameter_name, value in parameters_list:
            if parameter_name == 'parent_name' or parameter_name == 'process_name' \
                    or parameter_name == 'need_resampling':
                rows_number -= 1
                self.process_parameters_table_widget.setRowCount(rows_number)
                continue
            item = self.process_parameters_table_widget.horizontalHeaderItem(0)
            item.setText(_translate("LeafInterrogatorMainWindow", "Parameter"))
            item = QtWidgets.QTableWidgetItem()
            self.process_parameters_table_widget.setVerticalHeaderItem(row_counter, item)
            item = QtWidgets.QTableWidgetItem()
            self.process_parameters_table_widget.setItem(row_counter, 0, item)

            item = self.process_parameters_table_widget.verticalHeaderItem(row_counter)
            item.setText(_translate("LeafInterrogatorMainWindow", "New Row"))
            item = self.process_parameters_table_widget.item(row_counter, 0)
            item.setText(_translate("LeafInterrogatorMainWindow", str(parameter_name)))
            item = QtWidgets.QTableWidgetItem()
            self.process_parameters_table_widget.setItem(row_counter, 1, item)

            # item = self.process_parameters_table_widget.item(row_counter, 1)
            if type(value) is bool:
                self.param_combo_box = QtWidgets.QComboBox()
                self.param_combo_box.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
                self.param_combo_box.setObjectName("param_combo_box")
                self.param_combo_box.addItem("True")
                self.param_combo_box.addItem("False")

                self.process_parameters_table_widget.setCellWidget(row_counter, 1, self.param_combo_box)
            else:
                self._param_value_line_edit = QtWidgets.QLineEdit()
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(self._param_value_line_edit.sizePolicy().hasHeightForWidth())
                self._param_value_line_edit.setSizePolicy(sizePolicy)
                self._param_value_line_edit.setMaximumSize(QtCore.QSize(50, 16777215))
                self._param_value_line_edit.setInputMask("")
                # self._param_value_line_edit.setAlignment(QtCore.Qt.AlignCenter)
                self._param_value_line_edit.setObjectName("_param_value_line_edit")

                self._param_value_line_edit.setText(str(value))
                self.process_parameters_table_widget.setCellWidget(row_counter, 1, self._param_value_line_edit)

            row_counter += 1

    def build_result_table_widget(self, input_dict_list, process_name):
        _translate = QtCore.QCoreApplication.translate
        rows_number = len(input_dict_list)
        self.result_table_widget.setRowCount(rows_number)

        for row_counter, data_dict in enumerate(input_dict_list):
            result = data_dict[process_name]
            image_name = data_dict['image_name']
            item = self.result_table_widget.horizontalHeaderItem(0)
            item.setText(_translate("LeafInterrogatorMainWindow", str(process_name)))

            item = QtWidgets.QTableWidgetItem()
            self.result_table_widget.setVerticalHeaderItem(row_counter, item)

            item = QtWidgets.QTableWidgetItem()
            self.result_table_widget.setItem(row_counter, 0, item)

            # item = self.result_table_widget.verticalHeaderItem(row_counter)
            # item.setText(_translate("LeafInterrogatorMainWindow", "New Row"))

            item = self.result_table_widget.item(row_counter, 0)
            item.setText(str(image_name))

            item = QtWidgets.QTableWidgetItem()
            self.result_table_widget.setItem(row_counter, 1, item)

            item = self.result_table_widget.item(row_counter, 1)
            item.setText(str(result))

    def build_metadata_table_widget(self, num_rows=1):
        self.metadata_table_widget.setRowCount(num_rows)
        self.metadata_table_widget.setColumnCount(3)

        self.metadata_table_widget.setHorizontalHeaderLabels(["As class", "Parameter", "Value"])
        self.metadata_table_widget.verticalHeader().setVisible(False)
        self.metadata_table_widget.verticalHeader().setCascadingSectionResizes(True)
        self.metadata_table_widget.verticalHeader().setSortIndicatorShown(False)
        self.metadata_table_widget.horizontalHeader().setStretchLastSection(True)

    def add_row_metadata_table(self, num_rows):
        radio_button_widget = QtWidgets.QWidget()
        radio_button = QtWidgets.QRadioButton()
        self.button_group.addButton(radio_button)
        radio_button_layout = QtWidgets.QHBoxLayout(radio_button_widget)
        radio_button_layout.addWidget(radio_button)
        radio_button_layout.setAlignment(QtCore.Qt.AlignCenter)
        radio_button_layout.setContentsMargins(0, 0, 0, 0)
        radio_button_widget.setLayout(radio_button_layout)

        self.metadata_table_widget.setRowCount(num_rows)
        self.metadata_table_widget.setCellWidget(num_rows - 1, 0, radio_button_widget)

    def insert_to_metadata_table(self, metadata_dict):

        self.metadata_table_widget.setRowCount(len(metadata_dict))
        row_counter = 0
        sorted_keys = sorted(metadata_dict.keys())

        self.button_group = QtWidgets.QButtonGroup(self.metadata_table_widget)
        for key in sorted_keys:

            radio_button_widget = QtWidgets.QWidget()
            radio_button = QtWidgets.QRadioButton()
            self.button_group.addButton(radio_button)
            radio_button_layout = QtWidgets.QHBoxLayout(radio_button_widget)
            radio_button_layout.addWidget(radio_button)
            radio_button_layout.setAlignment(QtCore.Qt.AlignCenter)
            radio_button_layout.setContentsMargins(0, 0, 0, 0)
            radio_button_widget.setLayout(radio_button_layout)

            if key == metadata_dict.get('as_class_param', None):
                radio_button.setChecked(True)
            else:
                radio_button.setChecked(False)

            self.metadata_table_widget.setCellWidget(row_counter, 0, radio_button_widget)

            value = metadata_dict[key]

            self.metadata_table_widget.setItem(row_counter, 1, QtWidgets.QTableWidgetItem(key))
            self.metadata_table_widget.setItem(row_counter, 2, QtWidgets.QTableWidgetItem(value))

            row_counter += 1

    # options for elliptical Fourier transform
    def add_number_of_harmunics_efd_to_view(self):
        self.number_of_harmonics_lbl.show()
        self._number_of_harmonics.show()

    def remove_number_of_harmunics_efd_from_view(self):
        self.number_of_harmonics_lbl.hide()
        self._number_of_harmonics.hide()

    def add_harmonic_coefficients_to_view(self):
        self.harmonic_coeffs_groupbox.show()

    def remove_harmonic_coefficients_from_view(self):
        self.harmonic_coeffs_groupbox.hide()

    def add_options_to_view(self):
        self.efd_normalize_check_box.show()
        self.efd_size_invariant_check_box.show()

    def remove_options_from_view(self):
        self.efd_normalize_check_box.hide()
        self.efd_size_invariant_check_box.hide()

    def add_classes_with_colors(self, class_colors_dict):
        _translate = QtCore.QCoreApplication.translate
        button_list = []
        if self.scroll_area_widget_contents_grid_layout.count() > 0:
            for i in reversed(range(self.scroll_area_widget_contents_grid_layout.count())):
                widget = self.scroll_area_widget_contents_grid_layout.takeAt(i).widget()
                if widget is not None:
                    # widget will be None if the item is a layout
                    widget.deleteLater()

        row_counter = 0
        for class_value, color in class_colors_dict.items():
            btn = QtWidgets.QPushButton()
            btn.resize(btn.minimumSizeHint())
            btn.setMaximumSize(QtCore.QSize(20, 20))
            btn.setObjectName(class_value)
            # btn.setText(class_value)
            btn.setStyleSheet("background-color: rgb({}, {}, {});".format(color[0] * 255,
                                                                          color[1] * 255,
                                                                          color[2] * 255))
            label = QtWidgets.QLabel()
            label.setText(_translate("LeafInterrogatorMainWindow", "Class {}".format(class_value)))
            self.scroll_area_widget_contents_grid_layout.addWidget(label, row_counter, 0, 1, 1, QtCore.Qt.AlignCenter)
            self.scroll_area_widget_contents_grid_layout.addWidget(btn, row_counter, 1, 1, 1, QtCore.Qt.AlignCenter)
            button_list.append(btn)
            row_counter += 1

        spacer_item = QtWidgets.QSpacerItem(20, 40,
                                            QtWidgets.QSizePolicy.Minimum,
                                            QtWidgets.QSizePolicy.Expanding)
        self.scroll_area_widget_contents_grid_layout.addItem(spacer_item, row_counter, 0, 1, 1)
        return button_list

    def setup_progress_bar(self):
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.statusBar.setObjectName("statusBar")
        self.setStatusBar(self.statusBar)

        self.statusBar.addPermanentWidget(self.progress_bar)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1)

    def edit_landmarks_to_main_toolbar(self):
        # self.edit_landmarks_toolbar_widget = QtWidgets.QWidget(self._mainToolBar)
        # self.widget_5.setObjectName("edit_landmarks_toolbar_widget")

        # self.horizontalLayout_lt = QtWidgets.QHBoxLayout(
        #     self.edit_landmarks_toolbar_widget)
        # self.horizontalLayout_lt.setContentsMargins(0, 0, 0, 0)
        # self.horizontalLayout_lt.setSpacing(6)
        # self.horizontalLayout_lt.setObjectName("horizontalLayout_lt")
        #
        # self._change_landmarks_btn = QtWidgets.QPushButton(self.edit_landmarks_toolbar_widget)
        # self.horizontalLayout_lt.addWidget(self._change_landmarks_btn)
        self.remove_all_toolbar_actions()
        self.shift_landmarks.setObjectName("action_shift_landmarks")
        self._mainToolBar.addAction(self.shift_landmarks)

        # self.position_y_axis.setObjectName("action_position_y_axis")
        # self._mainToolBar.addAction(self.position_y_axis)

    def remove_all_toolbar_actions(self):
        for action in self._mainToolBar.actions():
            self._mainToolBar.removeAction(action)

    def show_hide_shape_space_components_holder_widget(self, show=True):
        if not show:
            self.shape_space_components_holder_widget.hide()
        else:
            self.shape_space_components_holder_widget.show()

    def show_hide_lda_solver_options(self, show=False):
        if not show:
            self._lda_options_widget.hide()
        else:
            self._lda_options_widget.show()
