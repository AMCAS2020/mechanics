import atexit
import json
import logging
import logging.config
import os
import shutil
import tempfile
import traceback
from numbers import Number

MINIMAL_VERSION = True 
MINIMAL_PROCESSES = False

import math
import numpy as np
import qtawesome as qta
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (QOpenGLVersionProfile, QSurfaceFormat)

from .crop_images_worker import CropImagesWorkerThread
from .delete_feature_from_metadata_worker import DeleteFeatureFromMetadataWorkerThread
from .detect_contours import DetectContour
from .detect_contours_worker import DetectContoursWorkerThread
#from .exploredatamainwindow_controller import ExploreDataController
#from .extension_exploredatadialog_gui import ExtensionExploreDataDialog
from .extension_exportShapeSpaceDialog_gui import ExtensionExportShapeSpaceDialog
from .extension_leaf_interrogator_gui import ExtensionLeafInterrogator
from .extension_saveAllImagesDialog_gui import ExtensionSaveAllImageDialog
#from .find_leaflet_worker import FindLeafletWorkerThread
from .helper import Helper
from .load_project_worker import LoadProjectWorkerThread
from .metadata import Metadata
#from .morphographx_data_handler import MorphographxDataHandler
from .open_folder_worker import OpenFolderWorkerThread
from .open_images_worker import OpenImagesWorkerThread
from .perform_process_worker import PerformProcessWorkerThread
from .process_plugin import Plugin
from .procrustes_analysis_worker import ProcrustesAnalysisWorkerThread
from .progress_bar_worker import ProgressBarWorkerThread
from .resample_contour_worker import ResampleContourWorkerThread
from .revert_resample_contour_worker import RevertResampleContourWorker
from .save_all_worker import SaveAllWorkerThread
from .save_project_worker import SaveProjectWorkerThread
from .save_user_input_metadata_worker import SaveUserInputMetadataWorkerThread
from .shape_space_plot_worker import ShapeSpacePlotWorkerThread
#from .split_leaflets_worker import SplitLeafletWorkerThread
#from .update_working_contour_worker import UpdateWorkingContourWorkerThread


# setup logger
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)


class LeafInterrogatorController(QtWidgets.QMainWindow,
                                 QtWidgets.QWidget,
                                 ExtensionLeafInterrogator,
                                 ExtensionSaveAllImageDialog,
#                                 ExtensionExploreDataDialog,
                                 ExtensionExportShapeSpaceDialog,
                                 ):

    sig_abort_progressbar_worker = pyqtSignal()

    def __init__(self, versionprofile=None):
        super().__init__()
        self.versionprofile = versionprofile

        self.temp_directory = tempfile.mkdtemp()
        self.data_directory = '.'
        self.data_directory_name = "data_dir"
        self.cut_points_file_path = os.path.join(self.temp_directory,
                                                 'cut_points_info.pkl')
        print("temp dir= ", self.temp_directory)
        atexit.register(Helper.exit_handler, self.temp_directory)

        LeafInterrogatorController.setup_logging()
        self.logger = logging.getLogger(__name__)

        self.setup_progress_bar()
        # index of the current main tab
        self.current_main_tab = 0
        # index of the current editing tab
        self.current_editing_tab = 0
        self.list_of_images = []
        self.dict_of_images = {}
        self.edit_cnt_dict_of_images = {}   # contains only founded contours
        self.dict_of_leaflets = {}  # {leaf_name: [leaflet1, leaflet2,...]}
        self.list_of_images_which_contours_not_founded = []
        self.cut_points_dict = {}  # {'terminal': [], 'leaflets': []}
        self.dict_of_cut_points_dict = {}  # {'image_name': {}, 'image_name':{},...}
        self.image_name_cut_points_mapping_dict = {}
        self.branch_path_leaflets_dict = {}
        self.image_name_main_rachis_dict = {}

        self._workers_done = 0
        self._threads = []

        self.image_number_to_name_map = {}
        self.images_dict = {}

        self.contours_dict_list = []
        self.image_name_cnt_color_map_list = {}

        self.total_images_in_folder = len(self.dict_of_images)
        self.current_image_number = 0
        self.edit_cnt_current_image_number = 0
        # for zoom in and out action
        self.zoom_scale_factor = 1.0
        self.default_image_size = QtCore.QSize(400, 400)

        self.store_size = self.default_image_size

        self.image = None

        self.cnt_points = None

        self.selected_image_format = 'default'

        self.metadata_dict = dict.fromkeys(['image_width',
                                            'image_height',
                                            'image_name',
                                            'threshold',
                                            'minimum_perimeter',
                                            'contour_thickness',
                                            'convexhull_thickness',
                                            'scale(pixel/cm)',
                                            'image_counter',
                                            'leaf number',
                                            'landmarks',
                                            'as_class_param',
                                            'num_pruning_iter',
                                            'min_leaflet_length',
                                            'petiol_width'
                                            'leaflet_position',
                                            'leaflet_number',
                                            'num_left_leaflets',
                                            'num_right_leaflets'])
        self.user_metadata_dict = {}

        self.initialize_opengl_widget()
        self.setupUi(self)

        self.connect_actions()

        self.metadata_table_rows = 1
        self.build_metadata_table_widget()

        self.shape_space_obj =None
        self.shape_space_plot_worker = None
        # number of leaf classes
        self.class_values = None
        self.shape_class_colors_dict = None

        self.update_gui_for_selected_method()

        self.show_leaflet_cut_points = False

        self.current_leaflet_tab = 0

        self.old_dict_of_images = None

        self.minimize_variance = False
        self.flipped_contours_info_dict = {}

        self.progress_bar_worker = ProgressBarWorkerThread(1)
        # self.update_status_bar("Ready")
        self.process_data_dict_list = []
        self.process_queue = []

        self.process_enabled = False

        self.setAcceptDrops(True)

        self.edit_contour_opengl_widget.temp_directory = self.temp_directory
        self.contour_opengl_widget.temp_directory = self.temp_directory
        self.image_opengl_widget.temp_directory = self.temp_directory
        self.logger.info('App successfully Initialized')

    @staticmethod
    def setup_logging(default_path='logging_config.json',
                      default_level=logging.INFO,
                      env_key='LOG_CFG'):
        """
        setup logging configuration
        :param default_path:
        :param default_level:
        :param env_key:
        :return:
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        file_paths = []
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            # Workaround for OSx dragging and dropping
            for url in e.mimeData().urls():
                # if op_sys == 'Darwin':
                #     fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())

                file_path = str(url.toLocalFile())
                if os.path.isdir(file_path):
                    self.open_folder(file_path)

                if os.path.isfile(file_path) and (
                            file_path.endswith('tar.gz') or
                            file_path.endswith('gz')):
                    self.load_project(file_path)

                elif os.path.isfile(file_path):
                    # print("file name=",file_path)
                    file_paths.append(file_path)

            if file_paths:
                self.open_file(file_paths)

        else:
            e.ignore()

    def connect_actions(self):
        self.crop_images_btn.clicked[bool].connect(self.crop_images)
        self.detect_contours_btn.clicked[bool].connect(self.detect_contours)

        fa_magnify_plus = qta.icon('fa.search-plus')
        fa_magnify_minus = qta.icon('fa.search-minus')
        fa_rotate_anticlockwise = qta.icon('fa.rotate-left')
        fa_rotate_clockwise = qta.icon('fa.rotate-right')

        self.previous_image_btn.clicked[bool].connect(self.previous_image)
        self.next_image_btn.clicked[bool].connect(self.next_image)

        self.resample_btn.clicked[bool].connect(self.resample_contour)
        self.edit_cnt_next_btn.clicked[bool].connect(self.edit_contour_next_image)
        self.edit_cnt_previous_btn.clicked[bool].connect(
            self.edit_contour_previous_image)

        self.add_row_metadata_table_btn.clicked[bool].connect(self.add_row_to_metadata_table)
        self.remove_row_metadata_table_btn.clicked[bool].connect(self.remove_row_from_metadata_table)
        self.insert_metadata_btn.clicked[bool].connect(self.save_user_input_metadata)

        self.apply_plot_conf_btn.clicked[bool].connect(self.create_shape_space_plot)
        self.method_combo_box.activated.connect(self.update_gui_for_selected_method)

        self.edited_contour_save_changes_btn.clicked[bool].connect(self.save_edited_contour_or_landmarks)

        self.apply_plot_options_btn.clicked[bool].connect(self.apply_plot_options)
        if MINIMAL_VERSION == False :
            self.find_leaflets_btn.clicked[bool].connect(self.show_split_points)
            self.split_leaflets_btn.clicked[bool].connect(self.split_leaflets)

            self.apply_edited_cut_points_btn.clicked[bool].connect(self.apply_edited_leaflets_cut_points)

            self.select_cut_points_btn.clicked[bool].connect(self.manage_edit_cut_points_buttons)
            self.add_cut_points_btn.clicked[bool].connect(self.manage_edit_cut_points_buttons)
            self.remove_cut_points_btn.clicked[bool].connect(self.manage_edit_cut_points_buttons)
        self.export_components_btn.clicked[bool].connect(self.export_shape_space_components)
        if MINIMAL_VERSION == False :
            self.explore_data_btn.clicked[bool].connect(self.explore_data)

        self.delete_resampled_contour_btn.clicked[bool].connect(self.revert_resampled_contour)
        self.shift_landmarks_anticlockwise_btn.setIcon(fa_rotate_anticlockwise)
        # --- menu actions ------
        self.action_open.triggered.connect(self.open_file)
        self.action_open_folder.triggered.connect(self.open_folder)
        self.action_save.triggered.connect(self.save_one)
        self.action_save_all.triggered.connect(self.save_all)
        self.action_quit.triggered.connect(QtWidgets.qApp.quit)
        self.actionZoom_In.triggered.connect(self.zoom_in)
        self.actionZoom_out.triggered.connect(self.zoom_out)
        self.actionReset_to_default.triggered.connect(self.reset_to_default)

        self.actionRotate_Left.triggered.connect(lambda: self.rotate(
            'anticlockwise'))
        self.actionRotate_Left.setIcon(fa_rotate_anticlockwise)
        self.actionRotate_Right.triggered.connect(lambda: self.rotate(
            'clockwise'))
        self.actionRotate_Right.setIcon(fa_rotate_clockwise)
        # self.actionUndo.triggered.connect(self.undo)
        # self.actionRedo.triggered.connect(self.redo)
        self.actionDelete.triggered.connect(self.delete_image)

        self.action_load_project.triggered.connect(self.load_project)
        self.action_save_project.triggered.connect(self.save_project)

        self.action_ocr.triggered.connect(self.fill_metadata_ocr_tool)
        self.action_metadata_from_image_name.triggered.connect(
            self.fill_metadata_image_name)

        self.action_import_contours_csv_files.triggered.connect(self.import_mgx_contours)
        # --- Tool bar --------------------------
        # self.shift_landmarks.triggered.connect(self.change_suggested_landmarks_position)
        self.shift_landmarks_anticlockwise_btn.setEnabled(False)
        self.shift_landmarks_anticlockwise_btn.clicked[bool].connect(
            self.change_suggested_landmarks_position)
        # self.position_y_axis.triggered.connect(self.change_suggested_landmarks_position)

        # --- tab widget -----------------------------
        self.main_tabs.currentChanged.connect(self.tab_changed)
        self.editing_tabs.currentChanged.connect(self.editing_tab_changed)
        self.contour_leaflet_tab.currentChanged.connect(self.leaflet_tab_changed)
        # --- radio buttons -----------------------
        self.edit_contour_points_radiobtn.toggled.connect(
            self.manage_contour_points_radio_buttons)
        self.edit_landmarks_radiobtn.toggled.connect(
            self.manage_contour_points_radio_buttons)
        if MINIMAL_VERSION == False :
            self.edit_leaflets_cut_points_check_box.toggled.connect(
                self.manage_cut_points_check_box)

            self.x_axis_pcs_radio_btn.clicked[bool].connect(
                self.change_shape_space_components)
            self.x_axis_processes_radio_btn.clicked[bool].connect(
                self.change_shape_space_components)

            self.y_axis_pcs_radio_btn.clicked[bool].connect(
                self.change_shape_space_components)
            self.y_axis_processes_radio_btn.clicked[bool].connect(
                self.change_shape_space_components)

            self.z_axis_pcs_radio_btn.clicked[bool].connect(
                self.change_shape_space_components)
            self.z_axis_processes_radio_btn.clicked[bool].connect(
                self.change_shape_space_components)

        # ---- checkbox ---------------------------
        if MINIMAL_VERSION == False:
            self.keep_unedited_contour_checkbox.toggled.connect(
                self.keep_unedited_contour)
        self.show_suggested_landmarks_checkbox.toggled.connect(
            self.show_hide_landmarks)

        self.show_all_landmarks_checkbox.toggled.connect(
            self.show_hide_landmarks)
        if MINIMAL_VERSION == False:
            self.edit_cnt_align_points_to_center_checkbox.toggled.connect(
                self.align_to_center)
        else:
            self.edit_contour_opengl_widget.align_to_center = True

        if MINIMAL_VERSION == False:
            self.edit_cnt_project_all_contours_on_screen_checkbox.toggled.connect(
                self.project_all_contours)

        self.redefine_landmarks_checkbox.toggled.connect(self.redefine_landmarks)

        # self.show_centroid_check_box.toggled.connect(self.create_shape_space_plot)
        # self.show_convex_hull_check_box.toggled.connect(self.create_shape_space_plot)
        # self.show_std_ellipse_check_box.toggled.connect(self.create_shape_space_plot)
        if MINIMAL_VERSION == False :
            self.resample_whole_leaves_check_box.toggled.connect(self.choose_working_contours)
            self.resample_lateral_leaflets_check_box.toggled.connect(self.choose_working_contours)
            self.resample_terminal_leaflets_check_box.toggled.connect(self.choose_working_contours)

            self.edit_cnt_convex_hull_checkbox.clicked.connect(self.choose_working_contours)

            self.automatic_flipping_check_box.toggled.connect(self.edit_for_automatic_flipping)
            self.submit_total_landmarks_btn.clicked[bool].connect(
              self.user_change_number_of_landmarks)
        # ---- line edit --------------------
 

        self.image_counter.returnPressed.connect(self.update_current_image_number)
        self.edit_cnt_counter.returnPressed.connect(self.update_current_image_number)
        # registration
        self.registration_tree_widget.doubleClicked.connect(
            self.apply_registration)
        self.apply_registration_btn.clicked[bool].connect(
            self.apply_registration)

        # process tree
        self.processes_tree_widget.itemSelectionChanged.connect(
            self.load_parameters_of_selected)
        # start processes
        self.start_process_btn.clicked[bool].connect(self.start_process)
        self.processes_tree_widget.itemDoubleClicked.connect(
            self.start_process_double_clicked)

    def start_progressbar(self):
        founded_in_thread = False
        for thread_tuple in self._threads:
            if "thread_progress_bar" in thread_tuple[0].objectName():
                founded_in_thread = True
                thread = thread_tuple[0]
                thread.start()
                break

        if not founded_in_thread:
            self.progress_bar_worker = ProgressBarWorkerThread(1)
            thread = QThread()
            thread.setObjectName('thread_progress_bar')
            self._threads.append((thread, self.progress_bar_worker,
                                  self.progress_bar_worker.work))
            self.progress_bar_worker.moveToThread(thread)

            self.progress_bar_worker.sig_done.connect(self.change_bar)

            thread.started.connect(self.progress_bar_worker.work)
            thread.start()

    def end_progressbar(self):
        try:
            self.progress_bar_worker.work_done()
            self._workers_done += 1

            self.logger.info('Asking each worker to abort!')
            self.logger.debug('objects on threads are:', self._threads)
            for thread, _, _ in self._threads:
                thread.quit()  # this will quit
                thread.wait()  # wait for it to *actually* quit

        except Exception:
            self.logger.error('Error on finishing progressbar thread', exc_info=True)

    def move_to_thread(self, worker_object, after_done_func, thread_name, work=None):
        """
            move worker to the different thread
        :param worker_object: Object
            object of the worker class
        :param after_done_func: Object
            function or method that responsible for handling modifications after
            worker has done the Job.
        :param thread_name: String
            name of the thread
        :return:
        """
        self.start_progressbar()
        founded_in_thread = False

        for thread_tuple in self._threads:
            thread = thread_tuple[0]
            thread_work = thread_tuple[2]
            if worker_object in thread_tuple:
                if not work:
                    if thread_work != worker_object.work:
                        thread.started.disconnect()
                        thread.started.connect(worker_object.work)
                else:
                    if thread_work != worker_object.work:
                        thread.started.disconnect()
                        thread.started.connect(work)
                founded_in_thread = True

                thread.start()
                break

        if not founded_in_thread:
            thread = QThread()
            thread.setObjectName('thread_' + thread_name)
            # self._threads.append((thread, worker_object))
            worker_object.moveToThread(thread)

            worker_object.sig_done.connect(after_done_func)

            if not work:
                thread.started.connect(worker_object.work)
                self._threads.append((thread, worker_object, worker_object.work))
            else:
                thread.started.connect(work)
                self._threads.append((thread, worker_object, work))
            thread.start()

        self.logger.info("{} thread started".format(thread_name))

    def apply_registration(self):
        self.logger.info("Start registration.")
        self.edit_contour_opengl_widget.clear_screen()
        get_selected_list = self.registration_tree_widget.selectedItems()
        if len(get_selected_list) > 0:
            if 'Procrustes' == get_selected_list[0].text(0):
                return
            try:
                # self.resample_contour()

                resampled_contours_dict_list = self.get_all_resampled_contours()
                self.edit_contour_opengl_widget.prepare_resampled_contours(
                    resampled_contours_dict_list)

            except Exception:
                self.logger.error('Failed to prepare re-sampled contours', exc_info=True)

                self.critical_message("Can not perform the operation!\n"
                                      "Please make sure you resampled the "
                                      "contours before doing this process!")
                return
        try:
            self.selected_method = get_selected_list[0]
        except IndexError:
            self.critical_message("Please select a method!")
            pass
        resampled_contours_info_dict_list = \
            self.edit_contour_opengl_widget.resampled_contours_info_dict_list

        mapped_main_cnt = self.edit_contour_opengl_widget.mapped_main_cnt
        try:
            self.procrustes_worker = ProcrustesAnalysisWorkerThread(
                self, self.temp_directory, resampled_contours_info_dict_list,
                mapped_main_cnt, self.selected_method, self.minimize_variance)
        except Exception:
            self.logger.error("Can not create procrustes worker.", exc_info=True)
            return

        self.move_to_thread(self.procrustes_worker,
                            self.on_apply_registration_worker_done,
                            "procrustes_analysis_working")

    @pyqtSlot(int, str)
    def on_apply_registration_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Registration did not Finish successfully.")
            return

        self.logger.info("Updating widget to show the registration results...")
        # Draw them on the screen
        if 'Generalized Procrustes' in self.selected_method.text(0):
             self.edit_contour_opengl_widget.procrustes_result_dict_list = self.procrustes_worker.procrustes_result_dict_list
             self.edit_contour_opengl_widget.generalized_proc_mean_shape_dict = self.procrustes_worker.generalized_proc_mean_shape_dict
#            self.edit_contour_opengl_widget.procrustes = False
#            self.edit_contour_opengl_widget.generalized_procrustes = True
#            if self.minimize_variance:
#                # run again on to get and display the final result
#                self.flipped_contours_info_dict = \
#                    self.procrustes_worker.flipped_contours_info_dict
#
#                resampled_contours_info_dict_list = \
#                    self.procrustes_worker.new_resampled_contours_info_dict_list
#                self.edit_contour_opengl_widget.resampled_contours_info_dict_list = resampled_contours_info_dict_list
#
#                a, b = self.procrustes_worker.generalized_procrustes_analysis(
#                    resampled_contours_info_dict_list)
#
#                # for d in a:
#                #     print(d, d['result_image_path'], d['texture_width'] ,d.keys())
#                self.edit_contour_opengl_widget.procrustes_result_dict_list = a
#                self.edit_contour_opengl_widget.generalized_proc_mean_shape_dict = b
#            else:
#                self.edit_contour_opengl_widget.procrustes_result_dict_list = \
#                    self.procrustes_worker.procrustes_result_dict_list
#                self.edit_contour_opengl_widget.generalized_proc_mean_shape_dict = \
#                    self.procrustes_worker.generalized_proc_mean_shape_dict
        else:
            self.edit_contour_opengl_widget.procrustes = True
            self.edit_contour_opengl_widget.generalized_procrustes = False
            if self.minimize_variance:
                # run again on to get and display the final result
                self.flipped_contours_info_dict = \
                    self.procrustes_worker.flipped_contours_info_dict

                resampled_contours_info_dict_list = \
                    self.procrustes_worker.new_resampled_contours_info_dict_list

                mapped_main_cnt = self.edit_contour_opengl_widget.mapped_main_cnt

                main_cnt, other_cnts = self.procrustes_worker.procrustes_analysis(
                    mapped_main_cnt,
                    resampled_contours_info_dict_list)

                self.edit_contour_opengl_widget.mapped_main_cnt = main_cnt

                self.edit_contour_opengl_widget.procrustes_result_dict_list = other_cnts
            else:
                self.edit_contour_opengl_widget.mapped_main_cnt = \
                    self.procrustes_worker.main_cnt

                self.edit_contour_opengl_widget.procrustes_result_dict_list = \
                    self.procrustes_worker.other_cnts

        self.edit_contour_opengl_widget.save_procrustes_results(self.temp_directory)
        self.edit_contour_opengl_widget.update_opengl_widget()
        self.end_progressbar()
        self.logger.info("Registration Finished.")

    def choose_working_contours(self):
        if self.old_dict_of_images is None:
            self.old_dict_of_images = self.dict_of_images

        elif not self.resample_whole_leaves_check_box.isChecked() and \
                not self.resample_lateral_leaflets_check_box.isChecked() and \
                not self.resample_terminal_leaflets_check_box.isChecked():
            self.dict_of_images = self.old_dict_of_images
            # print("After self.old_dict_of_images=", self.old_dict_of_images)
            self.update_base_on_choose_working_contours(self.old_dict_of_images)
            return

        if 'whole_leaf' in self.sender().objectName() and \
                self.resample_whole_leaves_check_box.isChecked():
            self.resample_lateral_leaflets_check_box.setChecked(False)
            self.resample_terminal_leaflets_check_box.setChecked(False)
            self.selected_working_contours = 'leaf'

        if 'lateral' in self.sender().objectName() and \
                self.resample_lateral_leaflets_check_box.isChecked():
            self.resample_whole_leaves_check_box.setChecked(False)
            self.selected_working_contours = 'lateral'

        if 'lateral' in self.sender().objectName() and \
                not self.resample_lateral_leaflets_check_box.isChecked():
            self.resample_lateral_leaflets_check_box.setChecked(False)
            if self.resample_terminal_leaflets_check_box.isChecked():
                self.selected_working_contours = 'terminal'

        if 'terminal' in self.sender().objectName() and \
                self.resample_terminal_leaflets_check_box.isChecked():
            self.resample_whole_leaves_check_box.setChecked(False)
            self.selected_working_contours = 'terminal'

        if self.resample_lateral_leaflets_check_box.isChecked() and \
            self.resample_terminal_leaflets_check_box.isChecked():
            self.selected_working_contours = 'leaflets'

        if 'convex_hull' in self.sender().objectName() and \
            self.edit_cnt_convex_hull_checkbox.isChecked():
            self.resample_whole_leaves_check_box.setChecked(False)
            self.resample_lateral_leaflets_check_box.setChecked(False)
            self.resample_terminal_leaflets_check_box.setChecked(False)

            self.selected_working_contours = 'convex_hull'

        self.update_base_on_choose_working_contours()

    def update_base_on_choose_working_contours(self, dict_of_images=None):
        update_working_contours = UpdateWorkingContourWorkerThread(self, self.data_directory,
                                                             self.temp_directory,
                                                             dict_of_images)

        self.move_to_thread(update_working_contours,
                            self.on_update_working_contours_worker_done,
                            "update_working_contour")

    @pyqtSlot(int, str)
    def on_update_working_contours_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not update working contours successfully.")
            return
        self.load_image(self.get_current_image_path())
        self.load_result_image(self.get_current_result_image_path())

        self.update_metadata_table_widget()
        self.update_edit_contour_tab()

        self.end_progressbar()

    @pyqtSlot(int)
    def change_bar(self, data):
        if data == 0:
            self.progress_bar.setMaximum(data)
        elif data == 1:
            self.progress_bar.setMaximum(data)

    def find_leaflets(self):
        result_image = self.get_current_result_image_path()
        if not result_image:
            self.end_progressbar()
            return

        leaflet_image_name, _ = Helper.separate_file_name_and_extension(
            self.get_current_image_path(), keep_extension=False)

        # if leaflet_image_name not in self.list_of_images:
        #     self.critical_message("Please make sure you choose correct image!")
        #     self.end_progressbar()
        #     return
        if self.is_leaflet_image(self.get_current_image_path()):
            self.critical_message("Please make sure you choose correct image!")
            return

        min_leaflet_length = self.min_leaflet_length
        petiole_width = self.petiole_width
        pruning_num_iterations = self.pruning_num_iterations
        apply_to_all = self.apply_to_all_find_leaflets_check_box.isChecked()

        dict_of_not_splited_images = {}
        counter = 1
        for _, image_name in self.dict_of_images.items():
            if not self.is_image_split(image_name):
                dict_of_not_splited_images[counter] = image_name
                counter += 1

        self.flfwt = FindLeafletWorkerThread(dict_of_not_splited_images,
                                             self.current_image_number,
                                             pruning_num_iterations,
                                             min_leaflet_length, petiole_width,
                                             self.temp_directory, apply_to_all)

        self.move_to_thread(self.flfwt, self.on_find_leaflet_worker_done, "find leaflet")

    @pyqtSlot(int, str)
    def on_find_leaflet_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not find leaflets successfully!")
            return

        image_name, _ = Helper.separate_file_name_and_extension(
            self.get_current_image_path(), keep_extension=True)
        result_image = self.get_current_result_image_path()
        # data = Helper.read_metadata_from_csv(self.temp_directory, result_image)

        # for splitting the leaflets we need to keep track of image and leaflets path
        self.dict_of_cut_points_dict.update(self.flfwt.dict_of_cut_points_dict)
        self.branch_path_leaflets_dict.update(self.flfwt.branch_path_leaflets_dict)
        self.image_name_main_rachis_dict.update(self.flfwt.image_name_main_rachis_dict)
        # self.contour_opengl_widget.cut_points = leaflets_cut_points

        self.contour_opengl_widget.cut_points_dict = self.dict_of_cut_points_dict[image_name]
        self.contour_opengl_widget.image_name_cut_points_mapping_dict = self.dict_of_cut_points_dict
        self.contour_opengl_widget.image_name_main_rachis_dict = self.image_name_main_rachis_dict

        self.write_cut_points_info_to_file()

        self.contour_opengl_widget.map_cut_points_to_opengl()

        self.contour_opengl_widget.prepare_cut_points(self.temp_directory, image_name)

        self.end_progressbar()

    def show_split_points(self):
        self.find_leaflets()
        self.show_leaflet_cut_points = True

    def split_leaflets(self):
        """

        :return:
        """
        result_image = self.get_current_result_image_path()

        if not result_image:
            # self.end_progressbar()
            return

        image_name = os.path.basename(os.path.dirname(result_image))
        image_path = self.get_current_image_path()

        leaflet_image_name, _ = Helper.separate_file_name_and_extension(
            image_path, keep_extension=True)

        if self.is_leaflet_image(self.get_current_image_path()):
            self.critical_message("Please make sure you choose correct image!")
            return
        apply_to_all = self.apply_to_all_find_leaflets_check_box.isChecked()

        result_image = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_name)
        dict_of_not_splited_images = {}
        counter = 1
        for _, image_name in self.dict_of_images.items():
            if not self.is_image_split(image_name):
                dict_of_not_splited_images[counter] = image_name
                counter += 1

        slfwt = SplitLeafletWorkerThread(dict_of_not_splited_images, self.current_image_number,
                                         self.branch_path_leaflets_dict,
                                         self.dict_of_cut_points_dict, apply_to_all,
                                         self.threshold, self.temp_directory)

        # self.start_progressbar()
        self._split_workers_done = 0

        self.move_to_thread(slfwt, self.on_split_leaflet_worker_done, "3")

    @pyqtSlot(int, str)
    def on_split_leaflet_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not split leaflets successfully!")
            return

        for image_name, _ in self.dict_of_cut_points_dict.items():
            self.save_data_after_split_leaflets(image_name)

        self.update_after_change_number_of_images()
        # self.update_display_images()
        self.total_images_in_folder = len(self.dict_of_images)

        self.current_image_number = 1
        self.set_image_numbers_label(self.current_image_number,
                                     self.total_images_in_folder)

        self.load_image(self.get_current_image_path())
        self.load_result_image(self.get_current_result_image_path())

        self.update_metadata_table_widget()
        self.contour_opengl_widget.mapped_cut_points = []

        self.end_progressbar()

    def save_data_after_split_leaflets(self, image_name=None):
        self.logger.info('Start saving data after split the leaflets')
        try:
            if not image_name:
                image_path = self.get_current_image_path()
            else:
                image_path = Helper.get_image_path_from_image_name(
                    self.temp_directory, image_name, self.data_directory_name)

            split_folder = Helper.get_or_create_image_directory(self.temp_directory,
                                                                image_path, type='split',
                                                                flag='get')
        except:
            Helper.critical_message("Please make sure you clicked 'find leaflet' button")
            return

        images = Helper.get_list_of_images(split_folder)

        leaflet_image_name, _ = Helper.separate_file_name_and_extension(
            image_path, keep_extension=True)

        if self.is_leaflet_image(image_path):
            self.critical_message("Please make sure you choose correct image!")
            return

        self.show_leaflet_cut_points = False

        try:
            self.dict_of_images = {k: v for k, v in
                                   self.dict_of_images.items()
                                   if v != leaflet_image_name}
        except:
            print('Can not find image in the list')

        try:
            last_key = np.array(list(self.dict_of_images.keys())).max()
        except Exception as e:
            self.logger.error('Failed to get max of the images dicts', exc_info=True)
            self.logger.debug('dict_of_images=', self.dict_of_images)
            last_key = 0

        # collect all images again after split
        for image in Helper.get_list_of_images(self.data_directory):
            if self.is_image_cropped(image):
                # cropped_dir = Helper.get_or_create_image_directory(self.temp_directory,
                #                                                    image, type='cropped')
                for crop_image in Helper.get_all_cropped_images_of_an_image(self.temp_directory, image):
                    folder = Helper.get_or_create_image_directory(self.temp_directory,
                                                                  crop_image, type='split')
                    if os.path.isdir(folder):
                        self.dict_of_leaflets[image] = Helper.get_list_of_images(folder)

                        self.add_number_of_leaflets_to_metadata(image)
            else:
                folder = Helper.get_or_create_image_directory(self.temp_directory,
                                                              image, type='split')
                if os.path.isdir(folder):
                    self.dict_of_leaflets[image] = Helper.get_list_of_images(folder)

                    self.add_number_of_leaflets_to_metadata(image)

        counter = 0
        for image in images:
            counter += 1
            if image not in self.dict_of_images.values():
                self.dict_of_images[last_key + counter] = image

            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, image)

            # try:
            #     threshold = float(self.threshold.text())
            #     approx_factor = float(self.approx_factor.text())
            #     minimum_perimeter = float(self.min_perimeter.text())
            #     maximum_perimeter = float(self.max_perimeter.text())
            #     contour_thickness = int(self.contour_thickness.text())
            #     convexhull_thickness = int(self.convexhull_thickness.text())
            # except Exception as e:
            #     # print(e)
            #     self.critical_message("Input value Error! \n" + str(e))
            #     return

            num_pruning_iterations = self.pruning_num_iterations
            min_leaflet_length = self.min_leaflet_length
            petiol_width = self.petiole_width

            leaflet_label = Helper.get_leaflet_labels_from_name(image)
            # find landmarks
            contours = Helper.load_contours_from_csv(self.temp_directory, result_image_path)
            height, width = Helper.get_image_height_width(result_image_path)
            landmarks = Helper.find_landmarks_from_contour(contours, width, height)

            # landmarks = []
            # save metadata --------------------------------
            metadata_dict = {
                'threshold': self.threshold,
                #'approx_factor': self.approx_factor,
                'minimum_perimeter': self.minimum_perimeter,
                'contour_thickness': self.contour_thickness,
                'convexhull_thickness': self.contour_thickness,
                'num_pruning_iter': num_pruning_iterations,
                'min_leaflet_length': min_leaflet_length,
                'petiol_width': petiol_width,
                'leaflet_position': leaflet_label['leaflet_position'],
                'leaflet_number': leaflet_label['number'],
            }
            md = Metadata(self.temp_directory, self.result_image_path, **metadata_dict)
            md.save_metadata()

        self.logger.info('Data saved after split the leaflets.')

    def resample_contour(self):
        """
        find sample points when the button is clicked.
        :return:
        """
        self.logger.info('Start re-sampling the contour(s)')
        try:
            number_of_points = self.number_of_sample_points
        except ValueError:
            self.critical_message("Please enter a valid Integer for number "
                                  "of sample points! ")
            return
        image_path = self.get_current_image_path()
        result_image_path = self.get_current_result_image_path()
        if image_path is None or result_image_path is None:
            return
        if MINIMAL_VERSION == False:
            resample_contour_w = ResampleContourWorkerThread(number_of_points,
                                                             self.get_current_result_image_path(),
                                                             self.get_current_image_path(),
                                                             self.data_directory,
                                                                 self.temp_directory,
                                                             self.edit_cnt_dict_of_images,
                                                             self.edit_cnt_apply_to_all_checkbox.isChecked(),
                                                                 self.edit_cnt_convex_hull_checkbox.isChecked(),
                                                             self.flip_all_to_same_side_checkbox.isChecked(),
                                                             self.current_editing_tab)
        else:
            resample_contour_w = ResampleContourWorkerThread(number_of_points,
                                                             self.get_current_result_image_path(),
                                                             self.get_current_image_path(),
                                                             self.data_directory,
                                                                 self.temp_directory,
                                                             self.edit_cnt_dict_of_images,
                                                             self.edit_cnt_apply_to_all_checkbox.isChecked(),
                                                             False,
                                                             False,
                                                             self.current_editing_tab)


        # self.save_user_input_metadata()
        self.create_all_contours_dict_list()

        self.move_to_thread(resample_contour_w, self.on_resample_contour_worker_done, "resample")

    @pyqtSlot(int, str)
    def on_resample_contour_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            self.critical_message("Can not find any image! Please make "
                                  "sure that you find the contour of an "
                                  "image in 'Detect Contour' tab.")
            self.end_progressbar()
            self.logger.info('Contours did not re-sampled successfully!')
            return

        # resampled_points = Helper.load_resampled_from_csv(
        #     self.temp_directory, self.get_current_result_image_path())

        # height, width = Helper.get_image_height_width(self.get_current_result_image_path())
        # self.edit_contour_opengl_widget.prepare_resampeled_main_contour(
        #     resampled_points, width, height)
        #
        # self.edit_contour_opengl_widget.prepare_suggested_landmarks()
        self.update_edit_contour_tab()
        # self.edit_contour_opengl_widget.update_opengl_widget()

        self.end_progressbar()
        self.logger.info('Contours are re-sampled!')

    def apply_edited_leaflets_cut_points(self):
        self.cut_points_dict = self.contour_opengl_widget.cut_points_dict
        image_path = self.get_current_image_path()
        if image_path is None:
            return
        image_name, _ = Helper.separate_file_name_and_extension(
            image_path, keep_extension=True)

        self.dict_of_cut_points_dict[image_name] = self.contour_opengl_widget.cut_points_dict
        # self.dict_of_cut_points_dict[image_name] = self.cut_points_dict
        self.contour_opengl_widget.selected_cut_points = []
        self.contour_opengl_widget.line_points = []

        # update cut points file for edited points
        self.write_cut_points_info_to_file()

        self.logger.info('Edited leaflets cut points successfully applied.')

    def manage_edit_cut_points_buttons(self):
        if self.sender().text() == "Add":
            if self.contour_opengl_widget.add_button_state_cut_points:
                self.add_cut_points_btn.setChecked(False)
                self.contour_opengl_widget.add_button_state_cut_points = False
            else:
                self.add_cut_points_btn.setChecked(True)
                self.contour_opengl_widget.add_button_state_cut_points = True

            self.select_cut_points_btn.setChecked(False)

            self.contour_opengl_widget.select_button_state_cut_points = False

            # self.contour_opengl_widget.add_button_state_cut_points = True

        if self.sender().text() == "Select":
            if self.contour_opengl_widget.select_button_state_cut_points:
                self.select_cut_points_btn.setChecked(False)
                self.contour_opengl_widget.select_button_state_cut_points = False
            else:
                self.select_cut_points_btn.setChecked(True)
                self.contour_opengl_widget.select_button_state_cut_points = True

            self.add_cut_points_btn.setChecked(False)

            self.contour_opengl_widget.add_button_state_cut_points = False

            # self.contour_opengl_widget.select_button_state_cut_points = True

        if self.sender().text() == "Remove":
            self.select_cut_points_btn.setChecked(False)
            self.add_cut_points_btn.setChecked(False)
            self.contour_opengl_widget.select_button_state_cut_points = False
            self.contour_opengl_widget.add_button_state_cut_points = False
            try:
                self.contour_opengl_widget.remove_selected_cut_points()
                self.contour_opengl_widget.update_opengl_widget()
            except AttributeError as e:
                self.logger.error('Failed to remove selected cut points.', exc_info=True)
                return
        self.logger.info('Selected button for editing the cut points is:{}'.format(self.sender().text()))

    def start_process_double_clicked(self, item, column):
        self.start_process(selected_process=item)

    def start_process(self, selected_process=None, show_graph=True):
        self.show_process_result_graph = show_graph
        try:
            if not selected_process:
                selected_process = self.processes_tree_widget.selectedItems()[0]
            try:
                loaded_classes_dict = self.plugins.load_classes()
            except AttributeError:
                directory = os.path.join('.', 'leafi', 'processes_directory')
                if MINIMAL_PROCESSES == True:
                    directory = os.path.join('.', 'leafi', 'minimal_processes_directory')
                self.load_process_plugins(directory)

            process_class_name_dict = \
                self.plugins.create_dict_processname_and_classname()
            # self.logger.info('Start process on selected process: {}'.format(selected_process[0].text(0)))
        except Exception:
            self.logger.error('Can not get the selected process.', exc_info=True)
            return

        try:
            process_name = selected_process.text(0)
        except:
            process_name = selected_process

        if process_name in process_class_name_dict.keys():
            class_name = process_class_name_dict[process_name]

            load_selected = loaded_classes_dict[class_name]

            for irow in range(self.process_parameters_table_widget.rowCount()):
                if self.process_parameters_table_widget.item(irow, 0) is None:
                    break

                parameter_name = self.process_parameters_table_widget.item(irow, 0).text()
                # get the attribute type
                param_type = type(getattr(load_selected, parameter_name))

                try:
                    if param_type is bool:
                        parameter_value = self.process_parameters_table_widget.cellWidget(irow, 1).currentText()
                        if parameter_value == "True":
                            parameter_value = True
                        else:
                            parameter_value = False
                    else:
                        parameter_value = \
                            param_type(self.process_parameters_table_widget.cellWidget(irow, 1).text())
                except ValueError:
                    self.logger.error("{} type is expected!".format(param_type), exc_info=True)
                    return
                # update the class attributes
                setattr(load_selected, parameter_name, parameter_value)

            self.perform_process = PerformProcessWorkerThread(
                self, self.data_directory,
                self.temp_directory,
#                self.apply_process_to_all_contours_checkbox.isChecked(),
                True,
                self.dict_of_images,
                load_selected, process_name)

            self.move_to_thread(self.perform_process, self.on_process_worker_done,
                                "{} process".format(process_name))

    @pyqtSlot(int, str)
    def on_process_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            self.critical_message("Can not perform the operation!\n"
                                  "Please make sure you resampled the "
                                  "contours before doing this process!")
            self.end_progressbar()
            self.logger.info(msg)
            return
        self.build_result_table_widget(
            self.perform_process.results, self.perform_process.process_name)

        results_filename = '{}_results.csv'.format(self.perform_process.process_name)
        Helper.create_result_csv_file(self.temp_directory, self.perform_process.results,
                                      filename=results_filename)

        if not self.show_process_result_graph:
            # create color for classes
            if not self.shape_class_colors_dict:
                self.class_values = set([result['class_value'] for
                                         result in self.perform_process.results])
                if self.class_values:
                    self.shape_class_colors_dict = Helper.get_random_rgb_class_color(
                        self.class_values)

            data_res = []
            for result in self.perform_process.results:
                res = result[self.perform_process.process_name]
                data_res.append(res)
                # save_path = os.path.join(self.temp_directory, self.perform_process.process.text(0))

                if result['class_value']:
                    color = self.shape_class_colors_dict[result['class_value']]
                else:
                    color = result['color_rgb']

                if isinstance(res, Number):
                    self.process_data_dict_list.append(
                        {'data': res,
                            'color_rgb': color,
                            'image_name': '',
                            'image_counter': '',
                            'class_value': result['class_value'],
                            'process_name': self.perform_process.process_name
                        })

            nd_data = np.array(data_res).reshape(-1, 1)
            if self.perform_process.process_name == self.x_axis_combobox.currentText():
                self.shape_space_plot_worker.update_coordinates_based_on_process(
                    data=nd_data, axis='x')
            if self.perform_process.process_name == self.y_axis_combobox.currentText():
                self.shape_space_plot_worker.update_coordinates_based_on_process(
                    data=nd_data, axis='y')
            if self.perform_process.process_name == self.z_axis_combobox.currentText():
                self.shape_space_plot_worker.update_coordinates_based_on_process(
                    data=nd_data, axis='z')
            try:

                self.move_to_thread(self.shape_space_plot_worker,
                                    self.on_create_process_plot_worker_done,
                                    "update plot",
                                    work=self.shape_space_plot_worker.draw_coordinate_values_with_sig)

            except Exception:
                self.logger.error('Error on applying plot options.', exc_info=True)
                return
            self.process_queue.remove(self.perform_process.process_name)

            if len(self.process_queue) > 0:
                self.start_process(self.process_queue[0], show_graph=False)

            self.end_progressbar()

            self.logger.info("{} process finished.".format(self.perform_process.process_name))
            return

        # data_float = []
        data_dict_list = []

        # create color for classes
        if not self.shape_class_colors_dict:
            self.class_values = set([result['class_value'] for
                                     result in self.perform_process.results])
            if self.class_values:
                self.shape_class_colors_dict = Helper.get_random_rgb_class_color(
                    self.class_values)

        for result in self.perform_process.results:
            res = result[self.perform_process.process_name]
            save_path = os.path.join(self.temp_directory, self.perform_process.process_name)

            if result['class_value']:
                color = self.shape_class_colors_dict[result['class_value']]
            else:
                color = result['color_rgb']

            if isinstance(res, Number):
                # data_float.append(res)

                data_dict_list.append({'data': res,
                                     'color_rgb': color,
                                     'image_name': '',
                                     'image_counter': '',
                                     'class_value': result['class_value'],
                                     'process_name': self.perform_process.process_name
                                     })

            else:
                data_dict_list.append({'data': res,
                             'color_rgb': color,
                             'image_name': result['image_name'],
                             'image_counter': result['image_counter'],
                             'class_value': result['class_value'],
                             'process_name': self.perform_process.process_name
                             })

        try:
            self.perform_process.ls.draw_graph_on_widget(reference=self,
                                                         data_dict_list=data_dict_list,
                                                         save_path=save_path)
        except:
            self.logger.error("can not plot the results!", exc_info=True)

        self.end_progressbar()

        self.logger.info("{} process finished.".format(self.perform_process.process_name))

    def update_status_bar(self, text):
        self.statusBar.showMessage(text)

    def get_components(self):
        x = self.x_axis_combobox.currentText()
        y = self.y_axis_combobox.currentText()
        if MINIMAL_VERSION == False:
            z = self.z_axis_combobox.currentText()
        else:
            z = '-'
        x_index, y_index, z_index = [0, 1, 2]

        if x != '-' and (MINIMAL_VERSION==True or self.x_axis_pcs_radio_btn.isChecked()):
            # self.n_component += 1
            x_index = self.x_axis_combobox.currentIndex()

        if y != '-' and (MINIMAL_VERSION==True or  self.y_axis_pcs_radio_btn.isChecked()):
            # self.n_component += 1
            y_index = self.y_axis_combobox.currentIndex()

        if z != '-' and (MINIMAL_VERSION==True or  self.z_axis_pcs_radio_btn.isChecked()):
            # self.n_component += 1
            z_index = self.z_axis_combobox.currentIndex()

        if x == '-' and y == '-' and z == '-':
            components = [x_index, y_index]
        elif x == '-' or x == '':
            components = [y_index, z_index]
        elif y == '-' or y == '':
            components = [x_index, z_index]
        elif z == '-' or z == '':
            components = [x_index, y_index]
        else:
            components = [x_index, y_index, z_index]

        return components

    def export_shape_space_components(self):
        if self.shape_space_obj is None:
            self.logger.info('Cannot find shape space object!')
            return

        image_path = self.get_current_image_path()
        if image_path is None:
            return
        try:
            image_name, _ = Helper.separate_file_name_and_extension(
                image_path)
        except:
            return 1

        # open a popup in order to get the image format
        dialog = QtWidgets.QDialog()
        dialog.ui = ExtensionExportShapeSpaceDialog()
        dialog.ui.setupUi(dialog)

        if dialog.exec_():
            # delete the attributes of popup dialog
            selected_format = dialog.ui.shape_space_plot_output_format_combo_box.currentText()
        else:
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            selected_format = 'pdf'

        filters = "All files (*.*);;"
        selected_filter = "All files (*.*)"
        # options = QtWidgets.QFileDialog.ShowDirsOnly
        dialog = QtWidgets.QFileDialog()
        user_path = dialog.getSaveFileName(
            self, "Save image", '.', filters, selected_filter)[0]
        if not user_path:
            return

        self.shape_space_obj.export_shape_space_transformed_data_to_csv(os.path.dirname(user_path),
                                                                        os.path.basename(user_path))

        self.shape_space_obj.export_shape_space_components_to_csv(os.path.dirname(user_path),
                                                                  os.path.basename(user_path))

        self.shape_space_obj.export_pca_prepared_data(os.path.dirname(user_path),
                                                      os.path.basename(user_path))
        self.shape_space_plot_worker.export_plot(os.path.dirname(user_path),
                                                 os.path.basename(user_path),
                                                 selected_format)
        self.logger.info('Shape space components are exported.')

    def update_gui_for_selected_method(self):
        self.logger.info('Update GUI for selected method...')
        method = self.method_combo_box.currentText()
        if method == "Elliptical Fourier Descriptors":
            self.add_options_to_view()
            self.add_number_of_harmunics_efd_to_view()
            self.add_harmonic_coefficients_to_view()

            self.show_hide_lda_solver_options(show=False)

        elif method == "LDA on Fourier coefficient":
            self.add_options_to_view()
            self.add_number_of_harmunics_efd_to_view()
            self.add_harmonic_coefficients_to_view()

            self.show_hide_lda_solver_options(show=True)

        elif method == "LDA":
            self.show_hide_lda_solver_options(show=True)
            self.remove_number_of_harmunics_efd_from_view()
            self.remove_harmonic_coefficients_from_view()
            self.remove_options_from_view()
        else:
            self.show_hide_lda_solver_options(show=False)
            self.remove_number_of_harmunics_efd_from_view()
            self.remove_harmonic_coefficients_from_view()
            self.remove_options_from_view()

    def prepare_shape_space_plot(self):

        if not self.shape_space_plot_worker:
            self.shape_space_plot_worker = ShapeSpacePlotWorkerThread(_controller_obj=self)

        self.shape_space_plot_worker.prepare_shape_space_plot_worker()

    def apply_plot_options(self):
        self.logger.info('Start applying plot options...')
        # In outer section of code
        # pr = profile.Profile()
        # pr.enable()
        try:

            # self.prepare_shape_space_plot()
            self.move_to_thread(self.shape_space_plot_worker,
                                self.on_apply_plot_options_worker_done,
                                "update plot",
                                work=self.shape_space_plot_worker.draw_coordinate_values_with_sig)

            # self.logger.info('Plot is updated.')
        except Exception:
            self.logger.error('Error on applying plot options.', exc_info=True)
            return
        # pr.disable()
        # Back in outer section of code
        # pr.print_stats(sort="calls")

    @pyqtSlot(int, str)
    def on_apply_plot_options_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            # self.critical_message("Error on apply plot options!")
            self.logger.info("Plot is NOT updated!")
            return

        # self.shape_space_plot_worker.canvas_widget.draw()
        self.end_progressbar()

        self.logger.info('Plot is updated!')

    def create_shape_space_plot(self):
        self.logger.info('Start creating plot ...')
        # if self.shape_space_plot_worker:
        #     self.shape_space_plot_worker.update_shape_space_plot()
        # In outer section of code
        # self.pr = profile.Profile()
        # self.pr.enable()

        if not self.process_enabled:
            try:

                self.move_to_thread(self.shape_space_plot_worker,
                                    self.on_create_shape_space_plot_worker_done,
                                    "update plot",
                                    work=self.shape_space_plot_worker.update_shape_space_plot)

            except Exception:
                self.logger.error('Error on applying plot options.', exc_info=True)
                return
        else:
            self.apply_processes_in_shape_space_panel()
            # self.update_based_on_process()

    @pyqtSlot(int, str)
    def on_create_shape_space_plot_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Can not create shape space plot!")
            return

        self.shape_space_obj = self.shape_space_plot_worker.shape_space_obj
        self.add_color_pickers_to_gui()

        self.shape_space_plot_worker.update_gui_according_to_changes()

        self.end_progressbar()

        self.logger.info('Shape space plot is created.')

    @pyqtSlot(int, str)
    def on_create_process_plot_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Plot is NOT updated!")
            return
        self.shape_space_plot_worker.update_gui_according_to_changes()
        self.end_progressbar()

        self.logger.info('Shape space plot is updated.')

    def display_image_in_shape_space_graphic_view(self, image_path):
        self.create_scene_for_shape_space_image()
        self.shape_space_image_scene.clear()

        img = Image.open(image_path)
        w, h = img.size

        # self.imgQ = ImageQt.ImageQt(img)  # we need to hold reference to imgQ, or it will crash
        # a = QtGui.QImageReader(image_path)
        # QtGui.QImage()
        # qimage = QtGui.QImage(self.imgQ)

        pix_map = QtGui.QPixmap(image_path)

        # myScaledPixmap = pixMap.scaled(self._shape_space_image_lbl.size(), Qt.KeepAspectRatio)
        # self._shape_space_image_lbl.setPixmap(myScaledPixmap)

        self.shape_space_image_scene.addPixmap(pix_map)

        self.shape_space_image_graphic_view.fitInView(
            QtCore.QRectF(0, 0, w, h), Qt.KeepAspectRatio)

        self.shape_space_image_scene.update()

    def add_color_pickers_to_gui(self):
        button_list = self.add_classes_with_colors(
            self.shape_class_colors_dict)

        for btn in button_list:
            btn.clicked[bool].connect(self.color_picker)

    def color_picker(self):
        button = self.sender()
        color_picker_dialog = QtWidgets.QColorDialog()
        btn_color = button.palette().button().color()
        # my_color = btn_color.getRgbF()
        color = color_picker_dialog.getColor(btn_color, color_picker_dialog)

        if not color.isValid():
            return
        try:
            new_color = color.getRgbF()
            if new_color:
                self.shape_class_colors_dict[button.objectName()] = new_color[:-1]
                self.shape_space_plot_worker.prepare_color_for_shape_space()
                self.add_color_pickers_to_gui()
                self.prepare_shape_space_plot()
                self.shape_space_plot_worker.draw_coordinate_values()
                self.shape_space_plot_worker.canvas_widget.draw()
        except:
            self.logger.error('Error on color picker!', exc_info=True)
            pass

    def tab_changed(self, selected_index):

        if selected_index == 2:
            self.current_main_tab = 2

            self.update_edit_cnt_dict_of_images()
            self.prepare_shape_space_plot()

        elif selected_index == 0:
            self.current_main_tab = 0
            self.load_image(self.get_current_image_path())
            self.load_result_image(self.get_current_result_image_path())

            self.edit_contour_opengl_widget.clear_screen()

        elif selected_index == 1:
            self.current_main_tab = 1
            self.edit_contour_opengl_widget.temp_directory = self.temp_directory
            self.edit_contour_opengl_widget.selected_point = []
            self.edit_contour_opengl_widget.selected_points_for_removing = []
            try:
                if self.is_image_split(self.get_current_result_image_path()):
                    self.resample_terminal_leaflets_check_box.setEnabled(True)
                    self.resample_lateral_leaflets_check_box.setEnabled(True)
                    self.resample_whole_leaves_check_box.setEnabled(True)
            except TypeError:
                pass

            try:
                if self.edit_contour_points_radiobtn.isChecked():
                    self.edit_contour_opengl_widget.editing_mode = 1

                elif self.edit_landmarks_radiobtn.isChecked():
                    self.edit_contour_opengl_widget.editing_mode = 2
                if self.image is not None:
                    self.edit_contour_opengl_widget.main_image = self.image

                    self.update_edit_contour_tab()

            except Exception:
                self.logger.error('Change to Editing tab.', exc_info=True)
                pass
        self.update_status_bar('')
        self.logger.info('Current tab index is: {}'.format(selected_index))

    def update_edit_contour_tab(self):
        self.update_edit_cnt_dict_of_images()
        image_path = self.get_current_image_path()
        if image_path is None:
            return
        result_image_path = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_path)

        if MINIMAL_VERSION == False and self.edit_cnt_convex_hull_checkbox.isChecked():
            self.cnt_points = Helper.load_resampled_from_csv(self.temp_directory,
                                                             result_image_path)
            if not self.cnt_points:
                self.cnt_points = Helper.load_convexhull_from_csv(self.temp_directory,
                                                                  result_image_path)
            if not self.cnt_points:
                self.critical_message("Can not find the Convex Hull. Please \n"
                                      "make sure that you found them in first panel.")
                self.edit_cnt_convex_hull_checkbox.setChecked(False)

                self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                                result_image_path)
        else:
            self.cnt_points = Helper.load_resampled_from_csv(self.temp_directory,
                                                             result_image_path)
            if not self.cnt_points:
                self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                                result_image_path)

        self.edit_contour_opengl_widget.image_path = result_image_path
        height, width = Helper.get_image_height_width(self.get_current_result_image_path())
        self.edit_contour_opengl_widget.prepare_main_contour_points(
            self.cnt_points, width, height, self.get_current_image_path())

        self.edit_contour_opengl_widget.update_opengl_widget()

        # get current image for edit contour tab
        current_image_name = self.dict_of_images[self.current_image_number]
        for number, image in self.edit_cnt_dict_of_images.items():
            if image == current_image_name:
                self.edit_cnt_current_image_number = number

        self.edit_cnt_set_contour_numbers_label()

        image_name, ext = Helper.separate_file_name_and_extension(
            self.get_current_image_path())

        self.update_status_bar(image_name + ext)

        self.logger.info('Edit contour tab updated.')

    def editing_tab_changed(self, selected_index):
        self.current_editing_tab = selected_index
        self.edit_contour_opengl_widget.current_editing_tab_index = selected_index

        if selected_index == 2:
            directory = os.path.join('.', 'leafi', 'processes_directory')
            if MINIMAL_PROCESSES == True:
                directory = os.path.join('.', 'leafi', 'minimal_processes_directory')
            print("directory=", directory)
            self.load_process_plugins(directory)

        if selected_index == 0:
            self.edit_contour_opengl_widget.clear_screen()
            self.edit_contour_opengl_widget.selected_point = []
            self.edit_contour_opengl_widget.selected_points_for_removing = []

            try:
                if self.edit_contour_points_radiobtn.isChecked():
                    self.edit_contour_opengl_widget.editing_mode = 1

                elif self.edit_landmarks_radiobtn.isChecked():
                    self.edit_contour_opengl_widget.editing_mode = 2
                if self.image is not None:
                    self.edit_contour_opengl_widget.main_image = self.image

                    self.update_edit_contour_tab()

            except Exception:
                self.logger.error('Change to Editing tab.', exc_info=True)
                pass
        if selected_index == 1:
            self.registration_tree_widget.focusNextChild()

    def leaflet_tab_changed(self, selected_index):
        self.current_leaflet_tab = selected_index

        if selected_index == 1:

            try:
                image_path = self.get_current_image_path()
                if not image_path:
                    return
                image_name, ext = Helper.separate_file_name_and_extension(
                    image_path)

                if image_name + ext in self.dict_of_cut_points_dict.keys():
                    self.contour_opengl_widget.map_cut_points_to_opengl()
                    self.contour_opengl_widget.prepare_cut_points(self.temp_directory,
                                                                  image_name + ext)
            except Exception as e:
                self.logger.error('Leaflet tab changed.', exc_info=True)

    def load_process_plugins(self, directory):
        """
        Load plugins into the process tree widget in process tab.
        :param directory: address of the plugins directory
        :return:
        """
        self.plugins = Plugin(directory)
        plugins_dict = self.plugins.load_plugins()
        self.build_processes_tree_widget(plugins_dict)

        self.logger.info('process plugin is loaded.')

    def load_parameters_of_selected(self):
        selected_item = None
        try:
            selected_item = self.processes_tree_widget.selectedItems()[0].text(
                0)

            class_name = self.plugins.create_dict_processname_and_classname()[selected_item]
            loaded_selected_class = self.plugins.load_classes()[class_name]
            param_dict = self.plugins.get_all_class_parameters(loaded_selected_class)

            self.build_parameter_table_widget(param_dict)
            return param_dict
        except (KeyError, IndexError) as e:
            self.logger.error('Item {} not found'.format(selected_item), exc_info=True)
            pass

    def initialize_opengl_widget(self):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSamples(3)
        QSurfaceFormat.setDefaultFormat(fmt)

        self.vp = QOpenGLVersionProfile()
        self.vp.setVersion(3, 3)
        self.vp.setProfile(QSurfaceFormat.CoreProfile)

    def draw_scale_line(self):
        line_width = 0
        try:
            line_width = float(self.get_line_width.text())
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)

        self.glwidget.draw_line_with_width(line_width)
#        print(len(self.glwidget.VAOs_list))

    def crop_images(self):
        """
        This function will separate contours in images with multiple
        leaves in them into individual images (contours)
        :return:
        """

        try:
            threshold = self.threshold
            # minimum_area = float(self.min_area.text())
            approx_factor = 0 #self.approx_factor
            minimum_perimeter = self.min_perimeter
            maximum_perimeter = self.max_perimeter
            contour_thickness = self.contour_thickness
            convexhull_thickness = 0 #self.convexhull_thickness
        except Exception as e:
            # print(e)
            self.critical_message("Input value Error! \n" + str(e))
            return

        image_path = self.get_current_image_path()
        if image_path is None:
            return
        # path to the result image with extension
        result_image_path = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_path)
        # put cropped images of the result image to this folder
        result_crop_dir = Helper.get_or_create_image_directory(
            self.temp_directory, image_path, type='cropped', flag='create')
        remove_squares_checkbox = False
        approximation_checkbox = False
        if MINIMAL_VERSION == False :
            remove_squares_checkbox = self.remove_squares_checkbox.isChecked()
            approximation_checkbox = self.approximation_checkbox.isChecked()
        self.crop_image_worker = CropImagesWorkerThread(
            self.data_directory, self.temp_directory, image_path,
            result_image_path, result_crop_dir, threshold, minimum_perimeter,
            maximum_perimeter, approx_factor, contour_thickness,
            convexhull_thickness,approximation_checkbox ,
            remove_squares_checkbox,)

        self.move_to_thread(self.crop_image_worker, self.on_crop_image_worker_done, "crop_image")

    @pyqtSlot(int, str)
    def on_crop_image_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not crop images successfully!")
            return
        self.list_of_images = self.crop_image_worker.list_of_images
        self.update_display_images()

        self.end_progressbar()

        self.logger.info('Image is cropped.')

    def detect_contours(self):
        try:
            threshold = self.threshold
            approx_factor = 0 #self.approx_factor
            minimum_perimeter = self.min_perimeter
            maximum_perimeter = self.max_perimeter
            contour_thickness = self.contour_thickness
            convexhull_thickness = 0 #self.convexhull_thickness
        except Exception as e:
            # print(e)
            Helper.critical_message("Input value Error! \n" + str(e))
            return

        image_path = self.get_current_image_path()
        if not image_path:
            return
        image_name, _ = Helper.separate_file_name_and_extension(
            image_path, keep_extension=True)
        # delete the existing files
        # Helper.delete_image_and_files(self.temp_directory,
        #                               image_name)

        # path to the result image with extension
        result_image_path = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_path)
        remove_squares_checkbox = False
        approximation_checkbox = False
        draw_convexhull_checkbox = False
        draw_contour_checkbox = True
        if MINIMAL_VERSION == False :
            remove_squares_checkbox = self.remove_squares_checkbox.isChecked()
            approximation_checkbox = self.approximation_checkbox.isChecked()
            draw_convexhull_checkbox = False
            draw_contour_checkbox = self.draw_contour_checkbox.isChecked()

        self.dcwt = DetectContoursWorkerThread(
            self.data_directory, self.temp_directory, self.data_directory_name,
            image_path, result_image_path, self.dict_of_images, threshold, minimum_perimeter,
            maximum_perimeter, approx_factor, contour_thickness,
            convexhull_thickness, remove_squares_checkbox,
            approximation_checkbox,
            self.apply_to_all_images_checkbox.isChecked(),
            draw_contour_checkbox,
            draw_convexhull_checkbox)

        # self.start_progressbar()
        self._workers_done = 0

        self.move_to_thread(self.dcwt, self.on_detect_contours_worker_done, "detect_contours")

    @pyqtSlot(int, str)
    def on_detect_contours_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not detect contours successfully!")
            return
        self.list_of_images_which_contours_not_founded = \
            self.dcwt.list_of_images_which_contours_not_founded

        if self.dcwt.not_founded_contours_counter > 0:
            error_string = 'No contour found for "{}" images.\n'.format(
                len(self.list_of_images_which_contours_not_founded))
            for i, image in enumerate(self.list_of_images_which_contours_not_founded):
                error_string += '{}- {}\n'.format(i + 1, image)
            self.warning_message(error_string)
        self.update_edit_cnt_dict_of_images()
        # if project all contours are checked then we uncheck it, because
        # we applied process on images again!
        if MINIMAL_VERSION == False and self.edit_cnt_project_all_contours_on_screen_checkbox.isChecked():
            self.edit_cnt_project_all_contours_on_screen_checkbox \
                .setChecked(False)
            self.project_all_contours()
            self.edit_contour_opengl_widget.update_opengl_widget()

        self.populate_image_counter_in_meta_data()
        self.load_result_image(self.get_current_result_image_path())

        self.update_metadata_table_widget()

        self.end_progressbar()

        self.logger.info('detect contours finished.')

    def get_all_resampled_contours(self):
        resampled_contours_dict_list = []

        for key, image_name in self.edit_cnt_dict_of_images.items():
            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, image_name)

            # check if we flipped the contour
            if result_image_path in self.flipped_contours_info_dict.keys():
                resampled_contours_dict_list.append(
                    self.flipped_contours_info_dict[result_image_path])
                continue

            try:
                color = self.image_name_cnt_color_map_list[image_name]
            except KeyError:
                self.create_all_contours_dict_list()
                color = self.image_name_cnt_color_map_list[image_name]

            resampled_cnts = Helper.load_resampled_from_csv(self.temp_directory,
                                                            result_image_path)
            resampled_contours_dict_list.append({
                                            'resampled_contour': resampled_cnts,
                                            'image_path': result_image_path,
                                            'color_rgb': color,
                                            'image_name': image_name
                                            })
        self.flipped_contours_info_dict = {}

        return resampled_contours_dict_list

    def get_all_contours(self):
        contours_dict_list = []
        for image_name in self.edit_cnt_dict_of_images.values():
            color = self.image_name_cnt_color_map_list[image_name]

            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, image_name)
            cnts = Helper.load_contours_from_csv(self.temp_directory,
                                                 result_image_path)

            image_path = Helper.get_image_path_from_image_name(
                self.data_directory, image_name, self.data_directory_name)
            contours_dict_list.append({
                                      'contour': cnts,
                                      'image_path': image_path,
                                      'color_rgb': color,
                                      'image_name': image_name
                                      })

        return contours_dict_list

    def get_all_aligned_contours(self):
        aligned_contours_dict_list = []
        using_aligned = True
        using_resampled = False
        for key, image_name in self.edit_cnt_dict_of_images.items():
            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, image_name)
            try:
                color = self.image_name_cnt_color_map_list[image_name]
            except KeyError:
                self.create_all_contours_dict_list()
                color = self.image_name_cnt_color_map_list[image_name]

            aligned_cnts = Helper.load_aligned_contour_csv(
                self.temp_directory, result_image_path)

            if not aligned_cnts:
                using_resampled = True
                using_aligned = False
                aligned_cnts = Helper.load_resampled_from_csv(self.temp_directory, result_image_path)
#AR - this is a bit hacked, and really intended to allow the EFD to work properly
#                original_contour = Helper.load_contours_from_csv(self.temp_directory,
                                   #                         result_image_path)
#                landmarks = Helper.find_landmarks(aligned_cnts,
#                                                  temp_dir=self.temp_directory,
#                                                  image_path=result_image_path)[0]            
#                avg_pos = (landmarks[1]+landmarks[0])*0.5
#                p1 = landmarks[1]-avg_pos
#                p0 = landmarks[0]-avg_pos
#                theta = -(math.atan2(p0[0,1],p0[0,0]))+3.14159/2.0
#                new_resampled_points = aligned_cnts[0]-avg_pos
#                R = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
#                new_resampled_points = np.dot(new_resampled_points,R.T)
#                aligned_cnts[0] = new_resampled_points

            if not aligned_cnts:
                using_resampled = False
                aligned_cnts = Helper.load_contours_from_csv(
                    self.temp_directory, result_image_path)
            class_name, class_value = Helper.get_class_from_metadata(
                self.temp_directory, result_image_path)

            aligned_contours_dict_list.append({
                'aligned_contour': aligned_cnts,
                'image_path': result_image_path,
                'color_rgb': color,
                'image_name': image_name,
                'class_name': class_name,
                'class_value': class_value,
            })
        if using_aligned == False and using_resampled == False:
           print("Warning: No resampled or aligned contours, extracted contours are being used!")
        if using_resampled == True:
           print("Warning: No aligned contours, resampled contours are being used!")

        return aligned_contours_dict_list

    def cur_image(self):

        if not self.get_current_image_path():
            self.image_opengl_widget.clear_screen()
            return

            self.set_image_numbers_label(
                self.current_image_number, self.total_images_in_folder
            )
        if self.current_main_tab == 0:
            self.load_image(self.get_current_image_path())
            self.load_result_image(self.get_current_result_image_path())

        image_path = self.get_current_image_path()
        if image_path is None:
            return
        image_name, ext = Helper.separate_file_name_and_extension(
            image_path)

        # remove the selected points
        self.contour_opengl_widget.selected_cut_points = []
        self.contour_opengl_widget.mapped_ready_to_change_cut_points = []

        self.update_status_bar(image_name+ext)

        self.update_metadata_table_widget()

    def next_image(self):
        if not self.get_current_image_path():
            self.image_opengl_widget.clear_screen()
            return

        if self.current_image_number < self.total_images_in_folder:
            self.current_image_number += 1
            self.set_image_numbers_label(
                self.current_image_number, self.total_images_in_folder
            )
        else:
            self.current_image_number = 1
            self.set_image_numbers_label(
                self.current_image_number, self.total_images_in_folder
            )
        if self.current_main_tab == 0:
            self.load_image(self.get_current_image_path())
            self.load_result_image(self.get_current_result_image_path())

        image_path = self.get_current_image_path()
        if image_path is None:
            return
        image_name, ext = Helper.separate_file_name_and_extension(
            image_path)

        # remove the selected points
        self.contour_opengl_widget.selected_cut_points = []
        self.contour_opengl_widget.mapped_ready_to_change_cut_points = []

        self.update_status_bar(image_name+ext)

        self.update_metadata_table_widget()

    def previous_image(self):

        if not self.get_current_image_path():
            self.contour_opengl_widget.clear_screen()
            return

        if self.current_image_number > 1:
            self.current_image_number -= 1
            self.set_image_numbers_label(
                self.current_image_number, self.total_images_in_folder
            )
        else:
            self.current_image_number = self.total_images_in_folder
            self.set_image_numbers_label(
                self.current_image_number, self.total_images_in_folder
            )

        if self.current_main_tab == 0:
            self.load_image(self.get_current_image_path())
            self.load_result_image(self.get_current_result_image_path())

        image_path = self.get_current_image_path()
        if image_path is None:
            return
        image_name, ext = Helper.separate_file_name_and_extension(
            image_path)

        # remove the selected points
        self.contour_opengl_widget.selected_cut_points = []
        self.contour_opengl_widget.mapped_ready_to_change_cut_points = []

        self.update_status_bar(image_name+ext)

        self.update_metadata_table_widget()

    def edit_contour_next_image(self):
        self.next_image()
        # if there is only one image, do nothing
        if self.total_images_in_folder == 1:
            return
        # check if the current image is exist
        if not self.get_current_image_path():
            return

        if self.edit_cnt_current_image_number < len(self.edit_cnt_dict_of_images):
            self.edit_cnt_current_image_number += 1
        else:
            self.edit_cnt_current_image_number = 1

        self.edit_cnt_set_contour_numbers_label()

        image_name = self.edit_cnt_dict_of_images[self.edit_cnt_current_image_number]
        result_image_path = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_name)

        self.edit_contour_opengl_widget.selected_point = []
        self.edit_contour_opengl_widget.selected_points_for_removing = []
        self.edit_contour_opengl_widget.mapped_landmarks = []
        if MINIMAL_VERSION == False and self.edit_cnt_convex_hull_checkbox.isChecked():
            self.cnt_points = Helper.load_resampled_from_csv(self.temp_directory,
                                                             result_image_path)
            if not self.cnt_points:
                self.cnt_points = Helper.load_convexhull_from_csv(self.temp_directory,
                                                                 result_image_path)
            if not self.cnt_points:
                self.critical_message("Can not find the Convex Hull. Please \n"
                                      "make sure that you found them in first panel.")
                self.edit_cnt_convex_hull_checkbox.setChecked(False)

                self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                                result_image_path)
        else:
            self.cnt_points = Helper.load_resampled_from_csv(self.temp_directory,
                                                             result_image_path)
            if not self.cnt_points:
                self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                                result_image_path)

        if self.cnt_points:
            self.edit_contour_opengl_widget.image_path = result_image_path
            height, width = Helper.get_image_height_width(self.get_current_result_image_path())
            self.edit_contour_opengl_widget.prepare_main_contour_points(
                self.cnt_points, width, height, self.get_current_image_path())

            if MINIMAL_VERSION == False and self.edit_cnt_project_all_contours_on_screen_checkbox.isChecked():
                self.create_all_contours_dict_list()
                self.edit_contour_opengl_widget.prepare_multiple_contours_points(
                    self.contours_dict_list)

            if self.edit_contour_opengl_widget.reset_positions:
                self.reset_to_default()
            if MINIMAL_VERSION == False and self.edit_cnt_align_points_to_center_checkbox.isChecked():
                self.align_to_center()

        # if self.edit_contour_opengl_widget.procrustes:
        #     self.apply_registration()

        self.edit_contour_opengl_widget.update_opengl_widget()

    def edit_contour_previous_image(self):
        self.previous_image()
        # if there is only one image, do nothing
        if self.total_images_in_folder == 1:
            return
        # check if the current image is exist
        if not self.get_current_image_path():
            return

        if self.edit_cnt_current_image_number > 1:
            self.edit_cnt_current_image_number -= 1
        else:
            self.edit_cnt_current_image_number = len(self.edit_cnt_dict_of_images)

        self.edit_cnt_set_contour_numbers_label()

        image_name = self.edit_cnt_dict_of_images[self.edit_cnt_current_image_number]
        result_image_path = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_name)

        self.edit_contour_opengl_widget.selected_point = []
        self.edit_contour_opengl_widget.selected_points_for_removing = []
        self.edit_contour_opengl_widget.mapped_landmarks = []

        if MINIMAL_VERSION == False and self.edit_cnt_convex_hull_checkbox.isChecked():
            self.cnt_points = Helper.load_resampled_from_csv(self.temp_directory,
                                                             result_image_path)
            if not self.cnt_points:
                self.cnt_points = Helper.load_convexhull_from_csv(self.temp_directory,
                                                                 result_image_path)
            if not self.cnt_points:
                self.critical_message("Can not find the Convex Hull. Please \n"
                                      "make sure that you found them in first panel.")
                self.edit_cnt_convex_hull_checkbox.setChecked(False)

                self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                                result_image_path)
        else:
            self.cnt_points = Helper.load_resampled_from_csv(self.temp_directory,
                                                             result_image_path)
            if not self.cnt_points:
                self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                                result_image_path)

        if self.cnt_points:
            self.edit_contour_opengl_widget.image_path = result_image_path
            height, width = Helper.get_image_height_width(self.get_current_result_image_path())
            self.edit_contour_opengl_widget.prepare_main_contour_points(
                self.cnt_points, width, height, self.get_current_image_path())

            if MINIMAL_VERSION == False and self.edit_cnt_project_all_contours_on_screen_checkbox.isChecked():
                self.create_all_contours_dict_list()
                self.edit_contour_opengl_widget.prepare_multiple_contours_points(
                    self.contours_dict_list)

            if self.edit_contour_opengl_widget.reset_positions:
                self.reset_to_default()
            if MINIMAL_VERSION == False and self.edit_cnt_align_points_to_center_checkbox.isChecked():
                self.align_to_center()

        # if self.edit_contour_opengl_widget.procrustes:
        #     self.apply_registration()

        self.edit_contour_opengl_widget.update_opengl_widget()

    def update_edit_cnt_dict_of_images(self):
        edit_contour_total_images = 0
        self.edit_cnt_dict_of_images = {}
        for image in self.dict_of_images.values():
            image_path = Helper.get_result_image_path_from_image_name(self.temp_directory, image)
            if os.path.isfile(image_path):
                edit_contour_total_images += 1
                self.edit_cnt_dict_of_images[edit_contour_total_images] = image

    def critical_message(self, error_message):
        reply = QtWidgets.QMessageBox.critical(self, "QMessageBox.critical()",
                                               error_message,
                                               QtWidgets.QMessageBox.Ok
                                               )
        # | QMessageBox.Retry | QMessageBox.Ignore)
        if reply == QtWidgets.QMessageBox.Ok:
            pass
            # elif reply == QMessageBox.Retry:
            #     self.criticalLabel.setText("Retry")
            # else:
            #     self.criticalLabel.setText("Ignore")

    def warning_message(self, error_message):
        reply = QtWidgets.QMessageBox.warning(self, "QMessageBox.warning()",
                                               error_message,
                                               QtWidgets.QMessageBox.Ok
                                               )
        if reply == QtWidgets.QMessageBox.Ok:
            pass

        return reply

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

    def get_current_image_path(self):
        """
        Get path of the current image.
        :return: string or None
        image path
        """
        try:
            image_name = self.dict_of_images[self.current_image_number]
        except KeyError as e:
            self.logger.error('Failed to get current image', exc_info=True)
            return None

        image_path = Helper.get_image_path_from_image_name(
            self.temp_directory, image_name, self.data_directory_name)

        if image_path is not None:
            if not os.path.isfile(image_path):
                return None

        return image_path

    def get_current_result_image_path(self):
        try:
            image_name = self.dict_of_images[self.current_image_number]
        except KeyError:
            self.logger.error('Failed to get current result image', exc_info=True)
            self.critical_message('can not find the result image!'
                                  'Please press "Apply" button in '
                                  'order to find contour of the '
                                  'loaded image!')
            return None

        result_image_path = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_name)

        if not result_image_path:
            return None

        if not os.path.isfile(result_image_path):
            return None

        return result_image_path

    def set_image_numbers_label(self, current_image_number,
                                total_images_in_folder):
        self.image_counter_label.setText(
            "{}".format(total_images_in_folder)
        )
        self.image_counter = current_image_number

    def zoom_in(self):
        if self.current_main_tab == 0:
            self.image_opengl_widget.zoom_in_opengl_widget(1.5)
            self.contour_opengl_widget.zoom_in_opengl_widget(1.5)

        elif self.current_main_tab == 1:
            self.edit_contour_opengl_widget.zoom_in_opengl_widget(1.5)

        self.edit_contour_opengl_widget.reset_positions = False

    def zoom_out(self):
        if self.current_main_tab == 0:
            self.image_opengl_widget.zoom_out_opengl_widget(0.75)
            self.contour_opengl_widget.zoom_out_opengl_widget(0.75)

        elif self.current_main_tab == 1:
            self.edit_contour_opengl_widget.zoom_out_opengl_widget(0.75)

        self.edit_contour_opengl_widget.reset_positions = False

    def reset_to_default(self):
        if self.current_main_tab == 0:
            self.image_opengl_widget.reset_positions = True
            self.image_opengl_widget.update_opengl_widget()        
        elif self.current_main_tab == 1:
            self.edit_contour_opengl_widget.reset_positions = True
            self.edit_contour_opengl_widget.update_opengl_widget()

    def load_image(self, image_path):
        """
        scale the input image in order to display it on the image_label
        :param image_path: input image path
        :return:
        """
        if not image_path:
            self.image_opengl_widget.clear_screen()
            return

        self.image_opengl_widget.create_rectangle_with_texture(image_path)
        self.image = self.image_opengl_widget.image

        # image_name, ext = Helper.separate_file_name_and_extension(
        #     image_path)
        # self.update_status_bar(image_name + ext)

        self.logger.debug('image "{}" is loaded.'.format(image_path))

    def load_result_image(self, image_path):
        if not image_path:
            self.contour_opengl_widget.clear_screen()
            return

        self.contour_opengl_widget.create_rectangle_with_texture(
            image_path)

        self.image = self.contour_opengl_widget.image

        image_name, ext = Helper.separate_file_name_and_extension(self.get_current_image_path())
        if image_name+ext in self.dict_of_cut_points_dict.keys():
            self.contour_opengl_widget.map_cut_points_to_opengl()
            self.contour_opengl_widget.prepare_cut_points(self.temp_directory,image_name + ext)

        self.logger.info('result image "{}" is loaded.'.format(image_path))

    def open_file(self, file_paths=None):
        self.list_of_images = []
        try:
            if not file_paths:
                file_paths = QtWidgets.QFileDialog.getOpenFileNames(
                    self, 'Open', '.')[0]
                if not file_paths:
                    return
            # build data directory
            self.data_directory = Helper.build_path(
                self.temp_directory, self.data_directory_name)
            if not os.path.exists(self.data_directory):
                os.mkdir(self.data_directory)

            self.open_image_worker = OpenImagesWorkerThread(
                self.data_directory, self.temp_directory, file_paths)

            self.move_to_thread(self.open_image_worker,
                                self.on_open_image_worker_done, "open_file")

        except Exception as e:
            self.logger.error('Failed to open image.', exc_info=True)
            return

    @pyqtSlot(int, str)
    def on_open_image_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Can not open image successfully!")
            return
        #print("list_of_images=", self.open_image_worker.list_of_images)
        self.list_of_images = self.open_image_worker.list_of_images

        self.update_display_images()

        self.end_progressbar()
        self.logger.info('images are opened.')

    def open_folder(self, folder_path=None):
        self.list_of_images = []
        try:
            if not folder_path:
                folder_path = QtWidgets.QFileDialog.getExistingDirectory(
                    self, 'Open', '.',
                    options=QtWidgets.QFileDialog.ShowDirsOnly)
                if not folder_path:
                    return
            # build data directory
            self.data_directory = Helper.build_path(
                self.temp_directory, self.data_directory_name)

            if not os.path.exists(self.data_directory):
                os.mkdir(self.data_directory)

            open_folder_worker = OpenFolderWorkerThread(
                self.data_directory, self.temp_directory, folder_path)

            self.move_to_thread(open_folder_worker, self.on_open_folder_worker_done, "open_folder")

            self.list_of_images = open_folder_worker.list_of_images

        except Exception as e:
            self.logger.error('Failed to open folder', exc_info=True)
            # tb = traceback.format_exc()
            # print(tb)
            return

    @pyqtSlot(int, str)
    def on_open_folder_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Can not open folder!")
            return
        self.update_display_images()

        self.end_progressbar()
        self.logger.info('Folder is opened.')

    def resize_images(self, list_of_images):
        temp_list = []
        resized_list = []
        for image_name in list_of_images:
            image_path = Helper.get_image_path_from_image_name(
                self.temp_directory, image_name, self.data_directory_name)

            image= DetectContour.fit_imagesize(image_path, 2500)

            if image is not None:
                image_name, _ = Helper.separate_file_name_and_extension(
                    image_path, True)
                resized_image_name = image_name
                new_image_path = os.path.join(self.data_directory,
                                              resized_image_name)

                temp_list.append(resized_image_name)
                resized_list.append(resized_image_name)
                DetectContour.save_resized_image(new_image_path, image)
                # os.remove(os.path.join(self.data_directory, image_name))
            else:
                temp_list.append(image_name)

        if len(resized_list) > 0 :
            warning_string = 'Image resolution > 2500 for "{}" images. Images have been resized\n'.format(
                    len(resized_list))
            for i, image in enumerate(resized_list):
                    warning_string += '{}- {}\n'.format(i + 1, image)
            self.warning_message(warning_string)
        self.logger.info('images are resized.')

        return temp_list

    def save_one(self):
        image_path = self.get_current_image_path()
        if image_path is None:
            return
        try:
            image_name, _ = Helper.separate_file_name_and_extension(
                image_path)
        except:
            return 1
        # directory = Helper.build_path(self.data_directory,
        #                               image_name)
        filters = "Images (*.png *.tiff *.jpg *.bmp *.jpeg);;"
        selected_filter = "Images (*.png *.tiff *.jpg *.bmp *.jpeg)"
        # options = QtWidgets.QFileDialog.ShowDirsOnly
        dialog = QtWidgets.QFileDialog()
        user_path = dialog.getSaveFileName(
            self, "Save image", '.', filters, selected_filter)[0]
        if not user_path:
            return

        # read the image using PIL
        image = Image.open(self.get_current_result_image_path())
        # save the image
        try:
            image.save(user_path)
        except:
            self.logger.error('Failed to save file in {} folder.'.format(user_path),
                              exc_info=True)
            QtWidgets.QMessageBox.warning(
                self, self.tr("Save Image"),
                self.tr("Failed to save file in the specified location."))
        # save contour result with the scale bar as an image
        self.contour_opengl_widget.save_scene_as_image(user_path)
        # copy the founded image contours in csv format from temp
        # directory to the user destination folder
        destination_folder = os.path.dirname(user_path)
        user_image_name, _ = Helper.separate_file_name_and_extension(user_path)

        temp_results_dir = Helper.get_result_image_directory(
            self.temp_directory, image_name)
        csv_files = Helper.get_list_of_csv_files(temp_results_dir)

        for file in csv_files:
            if image_name in file:
                full_file_name = Helper.build_path(temp_results_dir, file)
                shutil.copy2(full_file_name, destination_folder)
                if image_name != user_image_name:
                    old_file_name = Helper.build_path(destination_folder, file)
                    new_file_name = old_file_name.replace(image_name,
                                                          user_image_name)
                    os.rename(old_file_name, new_file_name)

        self.logger.info('image and its data saved.')

    def save_all(self):
        # open a popup in order to get the image format
        dialog = QtWidgets.QDialog()
        dialog.ui = ExtensionSaveAllImageDialog()
        dialog.ui.setupUi(dialog)
        dialog.exec_()
        # get the selected image format index
        selected_index = dialog.ui.selected_index_combobox
        self.selected_image_format = self.image_type_option(selected_index)
        image_base_name = dialog.ui.all_images_base_name.text()
        image_prefix = dialog.ui.all_images_prefix.text()
        image_suffix = dialog.ui.all_images_suffix.text()

        if dialog.ui.reject_button_clicked:
            # delete the attributes of popup dialog
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            return 1

        dialog = QtWidgets.QFileDialog()
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        destination_folder = dialog.getExistingDirectory(
            self, 'Save All', '.',
            options=QtWidgets.QFileDialog.ShowDirsOnly)
        if not destination_folder:
            return

        save_all_worker = SaveAllWorkerThread(
            self.data_directory, self.temp_directory,
            destination_folder, image_base_name,
            image_prefix, image_suffix, self.selected_image_format)

        self.move_to_thread(save_all_worker, self.on_save_all_worker_done, "save_all")

    @pyqtSlot(int, str)
    def on_save_all_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not save all successfully!")
            return
        self.end_progressbar()
        self.logger.info('All data are saved.')

    def save_project(self):
        filters = "gzip compression (*.gz);;bzip2 compression (*.bz2)"
        selected_filter = "gzip compression (*.gz)"
        dialog = QtWidgets.QFileDialog()
        destination_folder, compression_type = dialog.getSaveFileName(
            None, "Save image", '.', filters, selected_filter)
        if not destination_folder:
            return
        if not destination_folder.endswith('.tar.gz'):
            destination_folder = destination_folder + '.tar.gz'

        save_project_worker = SaveProjectWorkerThread(
            self.data_directory, self.temp_directory, destination_folder, compression_type)

        self.move_to_thread(save_project_worker, self.on_save_project_worker_done, "save_project")

    @pyqtSlot(int, str)
    def on_save_project_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not save project successfully!")
            return
        self.end_progressbar()
        self.logger.info('Project is successfully saved.')

    def load_project(self, file_path=None):
        try:
            if not file_path:
                filters = "gzip compression (*.gz);;bzip2 compression (*.bz2);;All files (*)"
                selected_filter = "gzip compression (*.gz)"
                dialog = QtWidgets.QFileDialog()
                dialog.setDefaultSuffix("*.gz")
                proj_file_path, _ = dialog.getOpenFileName(
                    None, 'Open', '.', filters, selected_filter)
                if not proj_file_path:
                    return
            else:
                proj_file_path = file_path
            # build data directory
            self.data_directory = Helper.build_path(
                self.temp_directory, self.data_directory_name)

            if not os.path.exists(self.data_directory):
                os.mkdir(self.data_directory)

            load_project_worker = LoadProjectWorkerThread(
                self, self.data_directory, self.temp_directory, proj_file_path)

            self.move_to_thread(load_project_worker, self.on_load_project_worker_done, "load_project")

        except Exception as e:
            self.logger.error('Failed to load project!', exc_info=True)
            # tb = traceback.format_exc()
            # print(tb)
            return

    @pyqtSlot(int, str)
    def on_load_project_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not load project successfully!")
            return
        self.update_display_images()
        image_name, ext = Helper.separate_file_name_and_extension(
            self.get_current_image_path())
        self.update_status_bar(image_name + ext)
        self.end_progressbar()

        self.logger.info('Project loaded.')

    def update_display_images(self, list_of_images=None):
        list_of_all_displaying_images = []
        list_of_new_images = []

        # register to delete resized files
        atexit.register(Helper.delete_resized_image, self.data_directory)
        if list_of_images is None:
            list_of_images = Helper.get_list_of_images(self.data_directory)

            list_of_images = self.resize_images(list_of_images)
            list_of_split = []
            list_of_cropped = []
            list_of_cropped_split = []
            list_of_new_images = []
            for image_name in list_of_images:
                if self.is_image_split(image_name):
                    #if self.current_leaflet_tab == 1:
                    image_split_dir = Helper.get_or_create_image_directory(
                        self.temp_directory, image_name, type='split')
                    temp = Helper.get_list_of_images(image_split_dir)
                    list_of_split.extend(temp)
                        # self.dict_of_leaflets[image_name] = []
                elif self.is_image_cropped(image_name):
                    #if self.current_leaflet_tab == 0:
                    image_cropped_dir = Helper.get_or_create_image_directory(
                        self.temp_directory, image_name, type='cropped')
                    list_of_cropped.extend(
                        Helper.get_list_of_images(image_cropped_dir))
                else:
                    list_of_new_images.append(image_name)

            for image_name in list_of_cropped:
                if self.is_image_cropped_and_split(image_name):
                    image_cropped_split_dir = Helper.get_or_create_image_directory(
                        self.temp_directory, image_name, type='cropped_split')

                    temp = Helper.get_list_of_images(image_cropped_split_dir)
                    list_of_cropped_split.extend(temp)
            # print("list_of_cropped_split=", list_of_cropped_split)
            # print("cropped=",list_of_cropped)
            # print("split=", list_of_split)
            list_of_all_displaying_images.extend(list_of_cropped_split)
            list_of_all_displaying_images.extend(list_of_cropped)
            list_of_all_displaying_images.extend(list_of_split)

            for img_cropped in list_of_cropped:
                for img_split in list_of_cropped_split:
                    img_name, _ = Helper.separate_file_name_and_extension(img_cropped)
                    if img_name in img_split:
                        try:
                            list_of_all_displaying_images.remove(img_cropped)
                            break
                        except:
                            break
        else:
            list_of_new_images = list_of_images

        self.dict_of_images = {}

        counter = 1
        for image_name in list_of_new_images:
            self.dict_of_images[counter] = image_name
            counter += 1

        for image_name in list_of_all_displaying_images:
            if image_name not in self.dict_of_images.values():
                self.dict_of_images[counter] = image_name
            counter += 1

        self.total_images_in_folder = len(self.dict_of_images)

        if self.current_image_number < 2:
            self.current_image_number = 1

        self.set_image_numbers_label(self.current_image_number,
                                     self.total_images_in_folder)

        # check if we have cut point information
        self.read_cut_points_info_from_file()

        self.load_image(self.get_current_image_path())
        self.load_result_image(self.get_current_result_image_path())

        self.update_metadata_table_widget()

        self.logger.info('Images are reordered and updated.')

    def is_image_split(self, image_name):
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)

        parent_image_name = Helper.get_parent_image_name(image_name)
        image_split_dir = Helper.get_or_create_image_directory(
            self.temp_directory, parent_image_name, type='split')

        if os.path.isdir(image_split_dir):
            self.logger.debug('image {} is split.'.format(image_name))
            return True
        else:
            self.logger.debug('image {} is NOT split.'.format(image_name))
            return False

    def is_image_cropped(self, image_name):
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)

        parent_image_name = Helper.get_parent_image_name(image_name)
        image_cropped_dir = Helper.get_or_create_image_directory(
            self.temp_directory, parent_image_name, type='cropped')

        if os.path.isdir(image_cropped_dir):
            self.logger.debug('image {} is cropped.'.format(image_name))
            return True
        else:
            self.logger.debug('image {} is NOT cropped.'.format(image_name))
            return False

    def is_image_cropped_and_split(self, image_name):
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)

        main_image_name = Helper.get_main_image_name(image_name)
        image_cropped_split_dir = Helper.get_or_create_image_directory(
            self.temp_directory, main_image_name, type='cropped_split')

        if os.path.isdir(image_cropped_split_dir):
            self.logger.debug('image {} is cropped and split.'.format(image_name))
            return True
        else:
            self.logger.debug('image {} is NOT cropped and split.'.format(image_name))
            return False

    def is_leaflet_image(self, image_name):
        """
        Check if the image is a leaflet image.
        :param image_name:
        :return:
        """
        leaflet_image_name, _ = Helper.separate_file_name_and_extension(
            image_name, keep_extension=True)

        if ('terminal' in leaflet_image_name) or \
            ('leaflet' in leaflet_image_name):
            self.logger.debug('image {} is a leaflet.'.format(image_name))
            return True
        else:
            self.logger.debug('image {} is NOT a leaflet.'.format(image_name))
            return False

    @staticmethod
    def image_type_option(index):
        value = {0: 'default',
                 1: 'png',
                 2: 'jpg',
                 3: 'bmp',
                 4: 'tiff',
                 5: 'jpeg'}.get(index, "default")
        return value

    def manage_contour_points_radio_buttons(self):
        if self.edit_contour_points_radiobtn.isChecked():
            if MINIMAL_VERSION==False:
                self.keep_unedited_contour_checkbox.setEnabled(True)
            self.show_suggested_landmarks_checkbox.setEnabled(False)
            self.show_all_landmarks_checkbox.setEnabled(False)
            self.redefine_landmarks_checkbox.setEnabled(False)
            self.shift_landmarks_anticlockwise_btn.setEnabled(False)
            self.edit_contour_opengl_widget.editing_mode = 1
            self.edit_contour_opengl_widget.mapped_landmarks = []
            self.edit_contour_opengl_widget.mapped_suggested_landmarks = []
            self.edit_contour_opengl_widget.mapped_all_landmarks = []

        elif self.edit_landmarks_radiobtn.isChecked():
            # self.edit_landmarks_to_main_toolbar()
            self.show_suggested_landmarks_checkbox.setEnabled(True)
            self.show_all_landmarks_checkbox.setEnabled(True)
            self.redefine_landmarks_checkbox.setEnabled(True)
            self.shift_landmarks_anticlockwise_btn.setEnabled(True)
            if MINIMAL_VERSION==False:
                self.keep_unedited_contour_checkbox.setEnabled(False)
                self.edit_contour_opengl_widget.total_number_of_landmarks = \
                    self.total_number_of_landmarks
            self.edit_contour_opengl_widget.editing_mode = 2
            self.edit_contour_opengl_widget.selected_point = None

        try:
            if self.image:
                height, width = Helper.get_image_height_width(self.get_current_result_image_path())
                self.cnt_points = Helper.load_contours_from_csv(self.temp_directory,
                                                                self.get_current_result_image_path())
                self.edit_contour_opengl_widget.prepare_main_contour_points(
                    self.cnt_points, width, height, self.get_current_image_path())
                self.edit_contour_opengl_widget.update_opengl_widget()
        except Exception:
            self.logger.error("Error in load main contour", exc_info=True)
            pass

    def manage_cut_points_check_box(self):
        if self.edit_leaflets_cut_points_check_box.isChecked():
            result_image = self.get_current_result_image_path()
            if result_image:
                height, width = Helper.get_image_height_width(result_image)
                contour_points = Helper.load_contours_from_csv(
                    self.temp_directory, result_image)
                self.contour_opengl_widget.prepare_contour_points(
                    contour_points, width, height)
            self.contour_opengl_widget.edit_leaflets_cut_points = True
            self.select_cut_points_btn.setEnabled(True)
            self.add_cut_points_btn.setEnabled(True)
            self.remove_cut_points_btn.setEnabled(True)
        else:
            self.select_cut_points_btn.setEnabled(False)
            self.add_cut_points_btn.setEnabled(False)
            self.remove_cut_points_btn.setEnabled(False)
            # remove the selected points
            try:
                self.contour_opengl_widget.edit_leaflets_cut_points = False
                self.contour_opengl_widget.selected_cut_points = []
                self.contour_opengl_widget.mapped_ready_to_change_cut_points = []
                self.contour_opengl_widget.update_opengl_widget()
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                pass

    def keep_unedited_contour(self):
        if MINIMAL_VERSION == True:
            return
        if self.keep_unedited_contour_checkbox.isChecked():
            self.edit_contour_opengl_widget.keep_unedited_cnt = True
            self.edit_contour_opengl_widget.update_opengl_widget()
        else:
            self.edit_contour_opengl_widget.keep_unedited_cnt = False
            self.edit_contour_opengl_widget.update_opengl_widget()

    def show_hide_landmarks(self):
        if self.show_suggested_landmarks_checkbox.isChecked():
            self.edit_contour_opengl_widget.show_suggested_landmarks = True
            try:
                self.edit_contour_opengl_widget.prepare_suggested_landmarks()
                self.edit_contour_opengl_widget.update_opengl_widget()
            except ValueError as e:
                print("ValueError:", e)
        elif not self.show_suggested_landmarks_checkbox.isChecked():
            self.edit_contour_opengl_widget.show_suggested_landmarks = False
            self.edit_contour_opengl_widget.update_opengl_widget()

        if self.show_all_landmarks_checkbox.isChecked():
            self.edit_contour_opengl_widget.show_all_landmarks = True
            try:
                self.edit_contour_opengl_widget.temp_directory = self.temp_directory
                self.edit_contour_opengl_widget.image_path = self.get_current_result_image_path()
                self.edit_contour_opengl_widget.prepare_all_landmarks()
                self.edit_contour_opengl_widget.update_opengl_widget()
            except ValueError as e:
                print("ValueError:", e)
        elif not self.show_all_landmarks_checkbox.isChecked():
            self.edit_contour_opengl_widget.show_all_landmarks = False
            self.edit_contour_opengl_widget.update_opengl_widget()
        # else:
        #     self.edit_contour_opengl_widget.show_suggested_landmarks = False
        #     self.edit_contour_opengl_widget.show_all_landmarks = False
        #     self.edit_contour_opengl_widget.update_opengl_widget()

    def user_change_number_of_landmarks(self):
        if self.redefine_landmarks_checkbox.isChecked():
            self.edit_contour_opengl_widget.redefine_landmarks = True

            self.edit_contour_opengl_widget.total_number_of_landmarks = \
                self.total_number_of_landmarks

    def redefine_landmarks(self):
        if self.redefine_landmarks_checkbox.isChecked():
            self.edit_contour_opengl_widget.redefine_landmarks = True
            self.show_all_landmarks_checkbox.setChecked(False)
            self.show_suggested_landmarks_checkbox.setChecked(False)
            if MINIMAL_VERSION == False:
                self.total_number_of_landmarks.setEnabled(True)

        else:
            self.edit_contour_opengl_widget.redefine_landmarks = False
            if MINIMAL_VERSION == False:
                self.total_number_of_landmarks.setEnabled(False)

    def edit_cnt_set_contour_numbers_label(self):
        self.edit_cnt_counter_label.setText(
            "{}".format(len(self.edit_cnt_dict_of_images))
        )
        self.edit_cnt_counter = self.edit_cnt_current_image_number

    def rotate(self, rd="anticlockwise"):
        # we just rotate the main image, if we rotate the result image
        # the founded contour and metadata will be wrong!!
        if rd == "anticlockwise":
            Helper.rotate_image(self.get_current_image_path(), -90)
            # Helper.rotate_image(self.get_current_result_image_path(), -90)
        else:
            Helper.rotate_image(self.get_current_image_path(), 90)
            # Helper.rotate_image(self.get_current_result_image_path(), 90)

        self.load_image(self.get_current_image_path())
        # self.load_result_image(self.get_current_result_image_path())
        # self.contour_opengl_widget.rotate(rotate_direction=rd)

    def align_to_center(self):
        if self.edit_cnt_align_points_to_center_checkbox.isChecked():
            self.edit_contour_opengl_widget.align_to_center = True
            self.edit_contour_opengl_widget.update_opengl_widget()
        else:
            self.edit_contour_opengl_widget.align_to_center = False
            self.edit_contour_opengl_widget.update_opengl_widget()

    def project_all_contours(self):
        if self.edit_cnt_project_all_contours_on_screen_checkbox.isChecked():
            self.edit_contour_opengl_widget.project_all_contours = True

            self.create_all_contours_dict_list()

            self.edit_contour_opengl_widget.prepare_multiple_contours_points(
                self.contours_dict_list)
            self.edit_contour_opengl_widget.update_opengl_widget()
        else:
            self.edit_contour_opengl_widget.project_all_contours = False
            self.edit_contour_opengl_widget.update_opengl_widget()

    def create_all_contours_dict_list(self):
        self.contours_dict_list = []
        for image_name in self.edit_cnt_dict_of_images.values():
            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, image_name)
            contour = Helper.load_contours_from_csv(self.temp_directory,
                                                    result_image_path)
            # if '_cropped_' in image_name:
            image_path = Helper.get_image_path_from_image_name(
                self.temp_directory, image_name, self.data_directory_name)

            # if not image_path:
            #     image_path = Helper.build_path(self.data_directory, image_name)

            if image_path == self.get_current_image_path():
                color = [1.0, 0.0, 0.0]
            else:
                color = Helper.get_random_rgb_color()

            self.contours_dict_list.append({
                'contour': contour,
                'image_path': image_path,
                'color_rgb': color,
                'image_name': image_name
                })

            self.image_name_cnt_color_map_list[image_name] = color

    def populate_image_counter_in_meta_data(self):
        for k, v in self.edit_cnt_dict_of_images.items():
            result_image_path = Helper.get_result_image_path_from_image_name(
                self.temp_directory, v)
            try:
                loaded_meta_dict = Helper.read_metadata_from_csv(
                    self.temp_directory, result_image_path)
            except FileNotFoundError:
                print('{} meta file not found!'.format(result_image_path))
                return
            if loaded_meta_dict['image_counter'] != str(k):
                loaded_meta_dict['image_counter'] = str(k)
                Helper.save_metadata_to_csv(self.temp_directory,
                                            result_image_path,
                                            loaded_meta_dict)

    def update_metadata_table_widget(self):
        current_result_image_path = self.get_current_result_image_path()

        if not current_result_image_path:
            return

        loaded_meta_dict = Helper.read_metadata_from_csv(
                self.temp_directory, current_result_image_path)

        if not loaded_meta_dict:
            return

        if loaded_meta_dict.get('image_counter') != self.current_image_number:
            loaded_meta_dict['image_counter'] = str(self.current_image_number)

            Helper.save_metadata_to_csv(self.temp_directory,
                                        current_result_image_path,
                                        loaded_meta_dict)

        self.metadata_table_rows = len(loaded_meta_dict)

        self.insert_to_metadata_table(loaded_meta_dict)

    def add_number_of_leaflets_to_metadata(self, image_name):
        folder = Helper.get_or_create_image_directory(
            self.temp_directory, image_name, type='split')

        leaflet_list = Helper.get_list_of_images(folder)

        num_left = 0
        num_right = 0
        num_terminal = 0

        for leaflet_name in leaflet_list:
            data = Helper.get_leaflet_labels_from_name(leaflet_name)
            if data['leaflet_position'] == 'right':
                num_right += 1
            elif data['leaflet_position'] == 'left':
                num_left += 1
            elif data['leaflet_position'] == 'terminal':
                num_terminal += 1

        result_image_name = Helper.get_result_image_path_from_image_name(
            self.temp_directory, image_name)

        loaded_meta_dict = Helper.read_metadata_from_csv(
            self.temp_directory, result_image_name)

        if not loaded_meta_dict:
            return
        loaded_meta_dict['num_left_leaflets'] = num_right
        loaded_meta_dict['num_right_leaflets'] = num_left
        loaded_meta_dict['num_terminal_leaflets'] = num_terminal

        Helper.save_metadata_to_csv(self.temp_directory,
                                    result_image_name,
                                    loaded_meta_dict)

    def add_row_to_metadata_table(self):
        try:
            self.metadata_table_rows += 1
            self.add_row_metadata_table(
                self.metadata_table_rows)
            # selected = self.metadata_table_widget.currentRow()
            # if self.apply_to_all_tables_checkbox.isChecked():
            #
            #     if selected >= 0:
            #         feature_name = self.metadata_table_widget.item(selected, 1).text()
            #         user_reply = self.question_message(
            #             "Do yo really want to DELETE '{}' "
            #             "row from metadata of ALL images?".format(feature_name))
            #
            #         dffm = DeleteFeatureFromMetadataWorkerThread(
            #             self, self.temp_directory,
            #             self.apply_to_all_tables_checkbox.isChecked(),
            #             user_reply)
            #         self.move_to_thread(dffm,
            #                             self.on_delete_feature_from_metadata_worker_done,
            #                             "delete_feature_from_metadata")
            #     else:
            #         self.warning_message("Pleas select a row to delete!")
        except AttributeError:
            self.metadata_table_rows -= 1
            self.warning_message("Please first load an image\n"
                                 "And find the contour!")
            return

    def remove_row_from_metadata_table(self):
        """
         Remove selected row from metadata information table.
        :return: -
        """
        selected = self.metadata_table_widget.currentRow()
        if self.apply_to_all_tables_checkbox.isChecked():

            if selected >= 0:
                feature_name = self.metadata_table_widget.item(selected, 1).text()
                user_reply = self.question_message(
                    "Do yo really want to DELETE '{}' "
                    "row from metadata of ALL images?".format(feature_name))

                dffm = DeleteFeatureFromMetadataWorkerThread(
                    self, self.temp_directory,
                    self.apply_to_all_tables_checkbox.isChecked(),
                    user_reply)
                self.move_to_thread(dffm,
                                    self.on_delete_feature_from_metadata_worker_done,
                                    "delete_feature_from_metadata")
            else:
                self.warning_message("Pleas select a row to delete!")
        else:
            if selected >= 0:
                feature_name = self.metadata_table_widget.item(selected, 1).text()
                user_reply = self.question_message("Do yo really want to DELETE '{}' row?".format(
                    feature_name))

                dffm = DeleteFeatureFromMetadataWorkerThread(
                    self, self.temp_directory,
                    self.apply_to_all_tables_checkbox.isChecked(),
                    user_reply)
                self.move_to_thread(dffm,
                                    self.on_delete_feature_from_metadata_worker_done,
                                    "delete_feature_from_metadata")

            else:
                self.warning_message("Pleas select a row to delete!")

    @pyqtSlot(int, str)
    def on_delete_feature_from_metadata_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not delete feature from metadata successfully!")
            return
        self.update_metadata_table_widget()
        # deselect the cell
        self.metadata_table_widget.setCurrentCell(-1, -1)
        self.end_progressbar()

        self.logger.info("The selected feature is removed from metadata.")

    def save_user_input_metadata(self):
        self.logger.info('Start save user metadata.')

        save_user_input_metadata = SaveUserInputMetadataWorkerThread(
            self, self.data_directory, self.temp_directory, self.dict_of_images,
            self.user_metadata_dict,
            self.apply_to_all_tables_empty_fields_checkbox.isChecked(),
            self.apply_to_all_tables_with_same_class_checkbox.isChecked(),
            False,#self.apply_to_all_tables_with_same_parent.isChecked(),
            self.apply_to_all_tables_checkbox.isChecked())

        self.move_to_thread(save_user_input_metadata,
                            self.on_save_user_input_metadata_worker_done,
                            "save_metadata")

    @pyqtSlot(int, str)
    def on_save_user_input_metadata_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not save user input successfully!")
            return
        self.update_metadata_table_widget()

        self.end_progressbar()

        self.logger.info('User input metadata saved.')

    def delete_image(self):
        """
        remove the image from the image dict and then update
        the view
        :return:
        """
        image_path = self.get_current_image_path()
        if image_path is None:
            return
        image_name, _ = Helper.separate_file_name_and_extension(
            image_path, keep_extension=True)
        if ('lateral' in image_name.lower()) or ('terminal' in image_name.lower()):
            temp = {}
            for k, v in self.dict_of_images.items():
                if 'terminal' in v:
                    if v.split('terminal')[0] in image_name:
                        continue
                elif 'lateral' in v:
                    if v.split('lateral')[0] in image_name:
                        continue
                temp[k] = v

            self.dict_of_images = temp
        else:
            self.dict_of_images = {k: v for k, v in
                                   self.dict_of_images.items()
                                   if v not in image_name}

        # delete the existing files
        Helper.delete_image_and_files(self.temp_directory,
                                      image_name)
        data_dir = Helper.get_image_path_from_image_name(
            self.temp_directory, image_name, self.data_directory_name)

        try:
            if not os.listdir(os.path.dirname(data_dir)):
                os.rmdir(os.path.dirname(data_dir))

            # delete the image from data_dir
            data_dir = Helper.build_path(
                self.temp_directory, 'data_dir', image_name)
            os.remove(data_dir)
        except FileNotFoundError:
            self.logger.error('Error on finding file to delete.', exc_info=True)
            pass

        # if 'lateral' in image_name or ('terminal' in image_name.lower()):
        #     result_image_path = Helper.get_result_image_path_from_image_name(
        #         self.temp_directory, image_name)
        #     result_directory = os.path.dirname(result_image_path)
        #     print("removed dir=", result_directory)
        #     try:
        #         shutil.rmtree(result_directory)
        #     except FileNotFoundError:
        #         tb = traceback.format_exc()
        #         print(tb)

        self.update_after_change_number_of_images()

        if self.current_main_tab == 1:
            self.update_current_image_number()

        self.logger.info('Image deleted.')

    def update_after_change_number_of_images(self):
        """
        update the view after change in number of images
        :return:
        """
        if not Helper.check_dict_validation(self.dict_of_images):
            self.dict_of_images = Helper.reorder_dict_keys(
                self.dict_of_images)

        self.total_images_in_folder = len(self.dict_of_images)
        if self.total_images_in_folder == 0:
            return

        if self.current_image_number > self.total_images_in_folder:
            self.current_image_number = 1

        if self.total_images_in_folder == 0:
            self.current_image_number = 0
            if len(Helper.get_list_of_images(self.data_directory)):
                self.update_display_images()
            else:
                self.edit_contour_opengl_widget.clear_screen()
                self.image_opengl_widget.clear_screen()
                self.contour_opengl_widget.clear_screen()

        else:
            self.update_display_images()
            self.load_image(self.get_current_image_path())

            self.load_result_image(self.get_current_result_image_path())

        self.set_image_numbers_label(self.current_image_number,
                                     self.total_images_in_folder)

        if self.current_main_tab == 1:
            self.edit_contour_opengl_widget.main_image = None
            self.edit_contour_opengl_widget.update_opengl_widget()

        self.update_metadata_table_widget()

        self.logger.info('UI updated after changing number of images.')

    def save_edited_contour_or_landmarks(self):
        result_image_path = self.get_current_result_image_path()
        if result_image_path is None:
            return
        self.edit_contour_opengl_widget.save_edited_contour_or_landmarks_results(
            self.temp_directory, result_image_path)

        if self.edit_contour_opengl_widget.editing_mode == 2:
            self.show_all_landmarks_checkbox.setChecked(True)
            self.redefine_landmarks_checkbox.setChecked(False)
            self.redefine_landmarks()
            self.edit_contour_opengl_widget.update_opengl_widget()

    def update_current_image_number(self):
        """
            update images based on the user input.
        :return:
        """
        if self.image is not None and self.current_main_tab == 1:
            self.current_image_number = int(self.edit_cnt_counter.text())
            self.image_counter = self.edit_cnt_counter.text()
            self.edit_cnt_current_image_number = int(self.edit_cnt_counter.text())
            self.edit_contour_opengl_widget.main_image = self.image
            self.update_edit_contour_tab()
        else:
            self.current_image_number = int(self.image_counter.text())
            self.update_after_change_number_of_images()

    def keyPressEvent(self, QKeyEvent):
        if self.current_main_tab == 0:
            if QKeyEvent.key() == Qt.Key_Backspace:
                print("delete")
                try:
                    self.delete_image()
                except TypeError:
                    pass

        if QKeyEvent.key() == Qt.Key_Left:
            self.previous_image()

        if QKeyEvent.key() == Qt.Key_Right:
            self.previous_image()

        # ==== edit leaflet cut points
        if self.current_leaflet_tab == 1 and self.edit_leaflets_cut_points_check_box.isChecked():
            if QKeyEvent.modifiers() & Qt.ControlModifier and QKeyEvent.key() == Qt.Key_Shift:
                self.add_cut_points_btn.setChecked(True)

            if QKeyEvent.modifiers() == Qt.ShiftModifier:
                self.select_cut_points_btn.setChecked(True)
        # ======

    def keyReleaseEvent(self, QKeyEvent):
        # ==== edit leaflet cut points
        if self.current_leaflet_tab == 1 and self.edit_leaflets_cut_points_check_box:
            if QKeyEvent.key() == (Qt.Key_Control | Qt.Key_Shift):
                self.add_cut_points_btn.setChecked(False)

            elif QKeyEvent.key() == Qt.Key_Shift:
                self.select_cut_points_btn.setChecked(False)
        # =======

    def fill_metadata_ocr_tool(self):
        regex = ''
        import pytesseract
        for key, image_name in self.dict_of_images.items():
            image_path = Helper.build_path(self.data_directory, image_name)

            # print(image_to_string(Image.open('Col x 2251_003_05.tif')))
            img = Image.open(image_path)
            result = pytesseract.image_to_string(img)

    def fill_metadata_image_name(self):
        pass

    def clear_program_completely(self):
        self.list_of_images = []
        self.dict_of_images = {}

        self.image_opengl_widget.clear_screen()
        self.contour_opengl_widget.clear_screen()
        self.edit_contour_opengl_widget.clear_screen()
        Helper.clear_data_and_temp_directories(
            self.data_directory, self.temp_directory)

    def explore_data(self):

        if self.shape_space_plot_worker.method == "Elliptical Fourier Descriptors":
            return

        pc_axis_dict = {}
        try:
            pc_axis_dict = {'x': self.x_axis_combobox,
                            'y': self.y_axis_combobox,
                            'components': self.get_components()}
        except Exception as e:
            self.logger.error('Failed to create axis dictionary', exc_info=True)

        try:
            edc = ExploreDataController(
                temp_dir=self.temp_directory,
                resampled_contours_info_dict_list=self.get_all_resampled_contours(),
                pc_axis_dict=pc_axis_dict,
                parent=self)

            # edc.setup_main_contour()
            if 'PCA' in self.method_combo_box.currentText():
                edc.shape_space_components = self.shape_space_obj.pca.components_
            # elif 'LDA' == self.method_combo_box.currentText():
            #     edc.shape_space_components = self.shape_space_obj.lda_clf.scalings_
            edc.percentage_of_variance = self.shape_space_obj.get_percentage_of_variance()
            edc.setup_main_contour()
            edc.show()

        except Exception:
            self.logger.error('Error on initializing the ExploreData panel', exc_info=True)
            return

    def write_cut_points_info_to_file(self):
        cut_points_info_dict = {}
        try:
            cut_points_info_dict = Helper.load_pickle_object(self.cut_points_file_path)
        except:
            pass

        cut_points_info_dict.update({'dict_of_cut_points_dict':
                                         self.dict_of_cut_points_dict,
                                     'image_name_main_rachis_dict':
                                         self.image_name_main_rachis_dict,
                                     'branch_path_leaflets_dict':
                                         self.branch_path_leaflets_dict
                                     })

        Helper.write_pickle_object(self.cut_points_file_path, cut_points_info_dict)
        self.resample_terminal_leaflets_check_box.setEnabled(True)
        self.resample_lateral_leaflets_check_box.setEnabled(True)
        self.resample_whole_leaves_check_box.setEnabled(True)

    def read_cut_points_info_from_file(self):
        try:
            cut_points_info_dict = Helper.load_pickle_object(self.cut_points_file_path)

            self.dict_of_cut_points_dict = cut_points_info_dict['dict_of_cut_points_dict']
            self.image_name_main_rachis_dict = cut_points_info_dict['image_name_main_rachis_dict']
            self.branch_path_leaflets_dict = cut_points_info_dict['branch_path_leaflets_dict']

            self.contour_opengl_widget.image_name_cut_points_mapping_dict = self.dict_of_cut_points_dict
            self.contour_opengl_widget.image_name_main_rachis_dict = self.image_name_main_rachis_dict

            self.resample_terminal_leaflets_check_box.setEnabled(True)
            self.resample_lateral_leaflets_check_box.setEnabled(True)
            self.resample_whole_leaves_check_box.setEnabled(True)

        except Exception as e:
#            print("There is no cut point information!", e) #Not relevant for minimal version
            pass

    def change_suggested_landmarks_position(self):
        resampled = None
        try:
            resampled = Helper.load_resampled_from_csv(
                self.temp_directory, self.get_current_result_image_path())
            if resampled:
                self.revert_resampled_contour()
        except TypeError:
            pass

            # self.critical_message("Please first delete the resampled contour!")
            # return

        # if self.sender().objectName() == 'action_shift_landmarks':
        #     _axis = 0
        # else:
        #     _axis = 1
        try:
            contours = Helper.load_contours_from_csv(
                self.temp_directory, self.get_current_result_image_path())
            old_landmarks = Helper.get_landmarks_from_meta_data(contours, self.temp_directory,
                                                                self.get_current_result_image_path())
            landmarks = Helper.find_landmarks_from_contour(
                contours, self.temp_directory, self.get_current_result_image_path(),
                old_landmarks=old_landmarks)
            metadata = Helper.read_metadata_from_csv(self.temp_directory, self.get_current_result_image_path())
            metadata['landmarks'] = landmarks.reshape(-1, 1, 2)
            Helper.save_metadata_to_csv(self.temp_directory, self.get_current_result_image_path(), metadata)
        except TypeError:
            return

        if resampled:
            self.resample_contour()

        self.update_metadata_table_widget()
        self.edit_contour_opengl_widget.prepare_all_landmarks()
        self.edit_contour_opengl_widget.update_opengl_widget()

    def revert_resampled_contour(self):
        # self.update_edit_contour_tab()
        self.logger.info('Start reverting the contour(s)')

        image_path = self.get_current_image_path()
        result_image_path = self.get_current_result_image_path()
        if image_path is None or result_image_path is None:
            return
        convex_hull_checkbox = False
        if MINIMAL_VERSION == False:
            convex_hull_checkbox = self.edit_cnt_convex_hull_checkbox.isChecked()
        revert_resample_cnt_w = RevertResampleContourWorker(
            self.get_current_result_image_path(),
            self.get_current_image_path(),
            self.data_directory,
            self.temp_directory,
            self.edit_cnt_dict_of_images,
            self.edit_cnt_apply_to_all_checkbox.isChecked(),
            convex_hull_checkbox,
            self.current_editing_tab)

        self.move_to_thread(revert_resample_cnt_w, self.on_revert_resample_contour_worker_done,
                            "revert_resampled")

    @pyqtSlot(int, str)
    def on_revert_resample_contour_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not revert resampled contour successfully!")
            return
        self.update_edit_contour_tab()

        self.end_progressbar()
        self.logger.info('Contours are reverted!')

    def edit_for_automatic_flipping(self):
        if self.automatic_flipping_check_box.isChecked():
            self.minimize_variance = True

        else:
            # make sure the dict is not contain any data
            self.minimize_variance = False
            self.flipped_contours_info_dict = {}

    def import_mgx_contours(self):
        self.list_of_cnt_csv_files = []
        try:
            filters = "csv files (*.csv);;txt files (*.txt);;All files (*)"
            selected_filter = "csv files (*.csv)"
            dialog = QtWidgets.QFileDialog()
            dialog.setDefaultSuffix("*.csv")
            file_paths, _ = dialog.getOpenFileNames(
                None, 'Open', '.', filters, selected_filter)
            if not file_paths:
                return

            # build data directory
            self.data_directory = Helper.build_path(
                self.temp_directory, self.data_directory_name)

            if not os.path.exists(self.data_directory):
                os.mkdir(self.data_directory)

            self.open_contour_files_worker = MorphographxDataHandler(
                self.data_directory, self.temp_directory, file_paths)

            self.move_to_thread(self.open_contour_files_worker,
                                self.on_open_contour_files_worker_done,
                                "open_files",
                                work=self.open_contour_files_worker.import_contour_csv_files)

        except Exception as e:
            self.logger.error('Failed to open files.', exc_info=True)
            return

    @pyqtSlot(int, str)
    def on_open_contour_files_worker_done(self, sig_id: int, msg: str):
        if sig_id == 0:
            if msg:
                self.critical_message(msg)
            self.end_progressbar()
            self.logger.info("Did not open contour files successfully!")
            return

        self.list_of_images = Helper.get_list_of_images(self.data_directory)
        self.update_display_images()

        self.end_progressbar()
        self.logger.info('files are opened.')

    def change_shape_space_components(self):
        sender_name = self.sender().objectName()
        if self.x_axis_pcs_radio_btn.isChecked() and 'x_axis_pcs_radio' in sender_name:
            # self.show_hide_shape_space_components_holder_widget(True)
            self.clear_shape_space_axis(axis='x')
            self.update_shape_space_axis_components(type='pc', axis='x')
        elif self.x_axis_processes_radio_btn.isChecked() and 'x_axis_processes_radio' in sender_name:
            # self.show_hide_shape_space_components_holder_widget(False)
            self.clear_shape_space_axis(axis='x')
            self.update_shape_space_axis_components(type='process', axis='x')

        if self.y_axis_pcs_radio_btn.isChecked() and 'y_axis_pcs_radio' in sender_name:
            self.clear_shape_space_axis(axis='y')
            self.update_shape_space_axis_components(type='pc', axis='y')
        elif self.y_axis_processes_radio_btn.isChecked() and 'y_axis_processes_radio' in sender_name:
            self.clear_shape_space_axis(axis='y')
            self.update_shape_space_axis_components(type='process', axis='y')

        if self.z_axis_pcs_radio_btn.isChecked() and 'z_axis_pcs_radio' in sender_name:
            self.clear_shape_space_axis(axis='z')
            self.update_shape_space_axis_components(type='pc', axis='z')
        elif self.z_axis_processes_radio_btn.isChecked() and 'z_axis_processes_radio' in sender_name:
            self.clear_shape_space_axis(axis='z')
            self.update_shape_space_axis_components(type='process', axis='z')

    def get_all_processes(self):
        directory = os.path.join('.', 'leafi', 'processes_directory')
        if MINIMAL_PROCESSES == True:
            directory = os.path.join('.', 'leafi', 'minimal_processes_directory')
            
        self.load_process_plugins(directory)

        process_class_name_dict = \
            self.plugins.create_dict_processname_and_classname()
        simple_processes = []
        for k, v in process_class_name_dict.items():
            loaded_selected_class = self.plugins.load_classes()[v]
            param_dict = dict(self.plugins.get_all_class_parameters(loaded_selected_class))
            if param_dict['parent_name'] == 'Basic Measures':
                simple_processes.append(k)

        return simple_processes

    def clear_shape_space_axis(self, axis='x'):
        if 1 < self.x_axis_combobox.count() and axis == 'x':
            self.x_axis_combobox.clear()
            self.x_axis_combobox.addItem("-")

        if 1 < self.y_axis_combobox.count() and axis == 'y':
            self.y_axis_combobox.clear()
            self.y_axis_combobox.addItem("-")

        if 1 < self.z_axis_combobox.count() and axis == 'z':
            self.z_axis_combobox.clear()
            self.z_axis_combobox.addItem("-")

    def update_shape_space_axis_components(self, type='pc', axis='x'):
        """

        :param type: Optional, 'pc' or 'process'
        :param axis: Optional, 'x', 'y', 'z'
        :return:
        """
        if type == 'process':
            processes_names = self.get_all_processes()
            if self.x_axis_processes_radio_btn.isChecked() or \
                    self.y_axis_processes_radio_btn.isChecked() or \
                    self.z_axis_processes_radio_btn.isChecked():
                # self.method_combo_box.setDisabled(True)
                self.process_enabled = True
        elif type == 'pc':
            if self.x_axis_pcs_radio_btn.isChecked() and \
                    self.y_axis_pcs_radio_btn.isChecked() \
                    and self.z_axis_pcs_radio_btn.isChecked():
                # self.method_combo_box.setEnabled(True)
                self.process_enabled = False
            try:
                if self.shape_space_plot_worker:
                    self.shape_space_plot_worker.init_tweaks()
            except AttributeError:
                return

        if axis == 'x':
            if type == 'process':
                for k in processes_names:
                    self.x_axis_combobox.addItem("{}".format(k))
        elif axis == 'y':
            if type == 'process':
                for k in processes_names:
                    self.y_axis_combobox.addItem("{}".format(k))
        elif axis == 'z':
            if type == 'process':
                for k in processes_names:
                    self.z_axis_combobox.addItem("{}".format(k))

        if self.x_axis_combobox.itemText(0) == '-':
            self.x_axis_combobox.removeItem(0)

        if self.y_axis_combobox.itemText(0) == '-':
            self.y_axis_combobox.removeItem(0)

        if self.z_axis_combobox.itemText(0) == '-':
            self.z_axis_combobox.removeItem(0)

        if self.x_axis_combobox.findText("-") == -1:
            self.x_axis_combobox.addItem("-")
        if self.y_axis_combobox.findText("-") == -1:
            self.y_axis_combobox.addItem("-")
        if self.z_axis_combobox.findText("-") == -1:
            self.z_axis_combobox.addItem("-")

        pervious_x_index = self.x_axis_combobox.currentIndex()
        pervious_y_index = pervious_x_index + 1

        self.x_axis_combobox.setCurrentIndex(pervious_x_index)

        self.y_axis_combobox.setCurrentIndex(pervious_y_index)
        if type == 'process' and axis == 'y':
            self.y_axis_combobox.setCurrentIndex(0)

        if type == 'process' and axis == 'z':
            self.z_axis_combobox.setCurrentIndex(0)
        else:
            self.z_axis_combobox.setCurrentIndex(
                self.z_axis_combobox.findText("-"))

    def apply_processes_in_shape_space_panel(self):
        self.logger.info("Apply processes in shape space panel ...")
        x_axis_process = self.x_axis_combobox.currentText()
        y_axis_process = self.y_axis_combobox.currentText()
        z_axis_process = self.z_axis_combobox.currentText()

        if 'PC' not in x_axis_process and x_axis_process != '-':
            self.process_queue.append(x_axis_process)
        if 'PC' not in y_axis_process and y_axis_process != '-':
            self.process_queue.append(y_axis_process)
        if 'PC' not in z_axis_process and z_axis_process != '-':
            self.process_queue.append(z_axis_process)

        # we start with first one and we will continue to calculate other
        # processes in work_done function by checking the self.process_queue
        if self.process_queue:
            self.start_process(self.process_queue[0], show_graph=False)

        elif x_axis_process == '-' or y_axis_process == '-' or z_axis_process == '-':
            self.process_enabled = False
            self.create_shape_space_plot()

# if __name__ == '__main__':
#
#     app = QtWidgets.QApplication(sys.argv)
#     app.installEventFilter(app)
#     LSA = LeafInterrogatorController()
#     LSA.show()
#
#     sys.exit(app.exec_())
