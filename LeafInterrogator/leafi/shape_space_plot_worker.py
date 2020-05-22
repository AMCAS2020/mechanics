import matplotlib

MINIMAL_VERSION = True

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import proj3d
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import numpy as np
import logging
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)
from .helper import Helper

from .shape_space import ShapeSpaceCalculator
from .detect_contours import DetectContour
#AR: To move into appropriate functions later
from sklearn import linear_model

class ShapeSpacePlotWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _controller_obj, parent=None, **kwargs):
        self.controller_object = _controller_obj
        super(ShapeSpacePlotWorkerThread, self).__init__(parent)

        prop_defaults = {
            "canvas_widget": None,
            "figure": None,
            "annotate_label": None,
            "annotate_image": None,
            "plot_type": '2d',
            "aligned_contours": None,

        }

        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        self.canvas_widget = prop_defaults['canvas_widget']
        self.figure = prop_defaults['figure']
        self.annotate_label = prop_defaults['annotate_label']
        self.annotate_image = prop_defaults['annotate_image']
        self.plot_type = prop_defaults['plot_type']

        self.all_artists = []
        self.annotation_images_scale = 4

        # self.aligned_contours = prop_defaults['aligned_contours']
        self.logger = logging.getLogger(__name__)

        self.x_axis_combobox = self.controller_object.x_axis_combobox
        self.y_axis_combobox = self.controller_object.y_axis_combobox
        if MINIMAL_VERSION == False:
            self.z_axis_combobox = self.controller_object.z_axis_combobox

        # self.plot_marker_size = self.controller_object.shape_space_plot_point_size

        self.method = self.controller_object.method_combo_box.currentText()

        self.coordinate_values = None
        self.shape_space_obj = None

        self.draw_annotation = True

        self.list_of_images = []

    def prepare_shape_space_plot_worker(self):
        self.logger.info('Prepare shape space plot for drawing...')
        if self.annotate_image:
            try:
                print("self.annotate_image=", self.annotate_image)
                self.annotate_image.remove()
            except ValueError:
                # self.logger.error('Remove annotate image.', exc_info=True)
                pass
            self.figure.canvas.draw()
        if self.annotate_label:
            try:
                print("self.annotate_label=", self.annotate_label)
                self.annotate_label.remove()
            except ValueError:
                # self.logger.error('Remove annotate label.', exc_info=True)
                pass
            self.figure.canvas.draw()

        # a figure instance to plot on
        if not self.figure:
            self.figure = plt.figure()

        else:
            self.figure.clf()

        if self.canvas_widget:
            self.controller_object.shape_space_tab_layout.removeWidget(self.canvas_widget)
        # this is the Canvas Widget that displays the `figure`
        self.canvas_widget = FigureCanvas(self.figure)
        self.controller_object.shape_space_tab_layout.addWidget(
            self.canvas_widget)

        self.figure.canvas.mpl_connect('button_press_event',
                                       self.on_click_matplot)
        # self.figure.canvas.mpl_connect('motion_notify_event',
        #                                self.on_hover_matplotlib)

        if self.controller_object.x_axis_combobox.findText("-") == -1:
            self.controller_object.x_axis_combobox.addItem("-")
            self.controller_object.y_axis_combobox.addItem("-")
            if MINIMAL_VERSION == False:
                self.controller_object.z_axis_combobox.addItem("-")

        self.logger.info('Shape space plot is prepared.')

    @pyqtSlot()
    def update_shape_space_plot(self):
        self.method = self.controller_object.method_combo_box.currentText()

        if self.method == "PCA":
            components = self.controller_object.get_components()
            self.shape_space_obj = ShapeSpaceCalculator(
                self.controller_object.get_all_aligned_contours(),
                _method=self.method,
                _n_component=None,
                _components=components)

            try:
                self.prepare_color_for_shape_space()

                self.shape_space_obj.prepare_vectors()
                self.shape_space_obj.map_class_to_color_and_value()
                self.shape_space_obj.compute_pca()
            except Exception as e:
                self.logger.error('Error on calculating PCA!', exc_info=True)
                self.sig_done.emit(0, 'Error on calculating PCA!')
                return
            try:
                self.coordinate_values = self.shape_space_obj.get_shape_space_coordinate_valuesSD()
                self.controller_object.x_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[0]
                self.controller_object.y_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[1]
            except IndexError:
                self.sig_done.emit(0, "Please make sure that you load at least\n"
                                      "2 images!")

                self.logger.error('Error on updating coordinates for PCA!', exc_info=True)

        elif self.method == "Elliptical Fourier Descriptors":
            components = self.controller_object.get_components()
            self.shape_space_obj = ShapeSpaceCalculator(
                self.controller_object.get_all_aligned_contours(),
                _method=self.method,
                _n_component=None,
                _components=components,
                # _a_n=self.controller_object.a_harmonic_coeff_check_box.isChecked(),
                # _b_n=self.controller_object.b_harmonic_coeff_check_box.isChecked(),
                # _c_n=self.controller_object.c_harmonic_coeff_check_box.isChecked(),
                # _d_n=self.controller_object.d_harmonic_coeff_check_box.isChecked(),
                _symmetric_components=self.controller_object.symmetric_components_check_box.isChecked(),
                _asymmetric_components=self.controller_object.asymmetric_components_check_box.isChecked(),
                normalize=self.controller_object.efd_normalize_check_box.isChecked(),
                size_invariant=self.controller_object.efd_size_invariant_check_box.isChecked())
            try:
                harmonics = self.controller_object.number_of_harmonics
            except ValueError:
                self.sig_done.emit(0, 'Error on get number of harmonics!')
                self.logger.error('Error on get number of harmonics!', exc_info=True)
                return

            try:
                self.prepare_color_for_shape_space()

                self.shape_space_obj.compute_efd(harmonics)
                self.shape_space_obj.map_class_to_color_and_value()
                self.shape_space_obj.compute_pca()
            except Exception as e:
                self.logger.error('Error on calculating EFD with PCA!', exc_info=True)
                self.sig_done.emit(0, 'Error on calculating EFD with PCA!')
                return
            try:
                self.coordinate_values = self.shape_space_obj.get_shape_space_coordinate_valuesSD()
                self.controller_object.x_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[0]
                self.controller_object.y_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[1]
            except IndexError:
                self.sig_done.emit(0, "Please make sure that you load at least\n"
                                      "2 images!")

                self.logger.error('Error on updating coordinates for EFD!', exc_info=True)

        elif self.method == "LDA":
            if self.controller_object.lda_eigen_solver_radio_btn.isChecked():
                solver = 'eigen'
            else:
                solver = 'svd'
            if self.controller_object.lda_svd_shrinkage_check_box.isChecked():
                shrinkage = True
            else:
                shrinkage = False
            components = self.controller_object.get_components()
            self.shape_space_obj = ShapeSpaceCalculator(
                self.controller_object.get_all_aligned_contours(),
                _method=self.method,
                _n_component=None,
                _components=components,
                lda_solver=solver,
                lda_shrinkage=shrinkage,
            )

            try:
                self.prepare_color_for_shape_space()

                self.shape_space_obj.prepare_vector_lda()
                self.shape_space_obj.map_class_to_color_and_value()
                self.shape_space_obj.compute_lda()
            except Exception as e:
                # self.sig_done.emit(1)
                self.logger.error('Error on calculating LDA!', exc_info=True)
                self.sig_done.emit(0, 'Error on calculating LDA!')
                return
            try:
                self.coordinate_values = self.shape_space_obj.get_shape_space_coordinate_valuesSD()
                print("percentage=", self.shape_space_obj.get_percentage_of_variance())
                self.controller_object.x_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[0]
                self.controller_object.y_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[1]
            except IndexError:
                self.sig_done.emit(0, "Please make sure that you load at least\n"
                                      "2 images!")

                self.logger.error('Error on updating coordinates for LDA!', exc_info=True)

        elif self.method == "LDA on Fourier coefficient":
            if self.controller_object.lda_eigen_solver_radio_btn.isChecked():
                solver = 'eigen'
            else:
                solver = 'svd'
            if self.controller_object.lda_svd_shrinkage_check_box.isChecked():
                shrinkage = True
            else:
                shrinkage = False
            components = self.controller_object.get_components()
            self.shape_space_obj = ShapeSpaceCalculator(
                self.controller_object.get_all_aligned_contours(),
                _method=self.method,
                _n_component=None,
                _components=components,
                # _a_n=self.controller_object.a_harmonic_coeff_check_box.isChecked(),
                # _b_n=self.controller_object.b_harmonic_coeff_check_box.isChecked(),
                # _c_n=self.controller_object.c_harmonic_coeff_check_box.isChecked(),
                # _d_n=self.controller_object.d_harmonic_coeff_check_box.isChecked(),
                _symmetric_components=self.controller_object.symmetric_components_check_box.isChecked(),
                _asymmetric_components=self.controller_object.asymmetric_components_check_box.isChecked(),
                normalize=self.controller_object.efd_normalize_check_box.isChecked(),
                size_invariant=self.controller_object.efd_size_invariant_check_box.isChecked(),
                lda_solver=solver,
                lda_shrinkage=shrinkage,
            )

            try:
                harmonics = self.controller_object.number_of_harmonics
            except ValueError:
                self.sig_done.emit(0, 'Error on get number of harmonics!')
                self.logger.error('Error on get number of harmonics!', exc_info=True)
                return

            try:
                self.prepare_color_for_shape_space()

                self.shape_space_obj.compute_efd(harmonics)
                self.shape_space_obj.map_class_to_color_and_value()
                self.shape_space_obj.compute_lda()
            except Exception as e:
                # self.sig_done.emit(1)
                self.logger.error('Error on calculating LDA!', exc_info=True)
                self.sig_done.emit(0, 'Error on calculating LDA!')
                return
            try:
                self.coordinate_values = self.shape_space_obj.get_shape_space_coordinate_valuesSD()
                self.controller_object.x_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[0]
                self.controller_object.y_axis_percentage_label = self.shape_space_obj.get_percentage_of_variance()[1]
            except:
                self.sig_done.emit(0, "Please make sure that you load at least\n"
                                      "2 images!")

                self.logger.error('Error on updating coordinates for LDA!', exc_info=True)

        self.sig_done.emit(1, "done")

    def update_coordinates_based_on_process(self, data, axis=None):
        """

        :param data: numpy ndarray(), shape is n by 1, 2 or 3
        :param axis: Optional, 'x', 'y', 'z',
        :return:
        """
        self.logger.info("Update coordinates based on process")
        if self.coordinate_values is not None:
            if not axis:
                self.coordinate_values = data
            elif 'x' in axis:
                self.coordinate_values[:, 0] = data[:, 0]
            elif 'y' in axis:
                if self.coordinate_values.shape[1] < 2:
                    self.coordinate_values = np.c_[self.coordinate_values, data]
                else:
                    self.coordinate_values[:, 1] = data[:, 0]
            elif 'z' in axis:
                if self.coordinate_values.shape[1] < 3:
                    self.coordinate_values = np.c_[self.coordinate_values, data]
                else:
                    self.coordinate_values[:, 2] = data[:, 0]
        else:
            self.coordinate_values = data

    def update_gui_according_to_changes(self):
        self.logger.info('Update plot GUI based on changes...')
        try:
            self.init_tweaks()
            self.prepare_shape_space_plot_worker()
            self.draw_coordinate_values()
        except Exception:
            self.logger.error('Error on update GUI based on changes!', exc_info=True)
            return

    @pyqtSlot()
    def draw_coordinate_values_with_sig(self):
        try:
            self.draw_coordinate_values()
        except Exception:
            self.logger.error('Error on draw coordinate GUI based on changes!', exc_info=True)
            self.sig_done.emit(0, 'Error on draw coordinate GUI based on changes!')
            return

        self.sig_done.emit(1, "done")

    def draw_coordinate_values(self):
        # self.prepare_shape_space_plot_worker()
        self.logger.info('Draw coordinate values...')
        general_lim = self.controller_object.plot_lim

        dict_of_class_value_and_point = \
            self.shape_space_obj.separate_points_by_class(self.coordinate_values)

        if MINIMAL_VERSION == False and self.coordinate_values.shape[1] >= 3 \
                and self.z_axis_combobox.currentText() != '-' \
                and self.controller_object.z_axis_pcs_radio_btn.isChecked():
            self.plot_type = "3d"
            self.axes = self.figure.add_subplot(111, projection='3d')

        elif self.coordinate_values.shape[1] >= 2:
            self.plot_type = "2d"
            self.axes = self.figure.add_subplot(111)

        if self.plot_type == "2d" and MINIMAL_VERSION==False and self.controller_object.z_axis_processes_radio_btn.isChecked():
            try:
                point_sizes = np.abs(self.coordinate_values[:, 2])
            except IndexError:
                self.logger.error("Can not get Point size form 'coordinate_values'! ")
                point_sizes = np.ones(shape=(self.coordinate_values.shape[0], 1))
        else:
            point_sizes = np.ones(shape=(self.coordinate_values.shape[0], 1))

        for class_value, points in dict_of_class_value_and_point.items():
            points = np.array(points, dtype='float32')
            color = self.shape_space_obj.get_class_color(class_value)
            if self.plot_type == "3d":
                self.axes.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color],
                                  s=self.controller_object.shape_space_plot_point_size * point_sizes)
            elif self.plot_type == "2d":
                self.axes.scatter(points[:, 0], points[:, 1], c=[color],
                                  s=self.controller_object.shape_space_plot_point_size) #*point_sizes )

                # check if there are enough points to build a convex hull
                if len(points) > 3:
                    if MINIMAL_VERSION == False and self.controller_object.show_convex_hull_check_box.isChecked():
                        # find and draw convex hull
                        hull = Helper.find_shape_space_convex_hull(points[:, :2])

                        for i, simplex in enumerate(hull.simplices):
                            self.axes.plot(points[simplex, 0], points[simplex, 1], c=color)

                    if self.controller_object.show_centroid_check_box.isChecked():
                        # find and draw centroid
                        centroid = Helper.compute_centroid_np(points)
                        self.axes.scatter(centroid[0, 0], centroid[0, 1], s=20 * 2 ** 2,
                                          c=[color], marker='x', edgecolor='k')

                    if self.controller_object.show_std_error_ellipse_check_box.isChecked():
                        # calculate and draw the std. deviation ellipse for each class
                        x = points[:, 0]
                        y = points[:, 1]
                        # n_std = 1
                        num_curves = self.controller_object.number_of_std_curves
                        ellipse_scale = self.controller_object.number_of_std
                        for i in range(1, num_curves + 1):
                            ellipse_scale += ellipse_scale
                            w, h, theta = Helper.calculate_std_err(x, y,ellipse_scale=ellipse_scale)

                            ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                                          width=w, height=h,
                                          angle=theta, color=color)
                            ell.set_facecolor('none')
                            self.axes.add_artist(ell)

                    if self.controller_object.show_std_ellipse_check_box.isChecked():
                        # calculate and draw the std. deviation ellipse for each class
                        x = points[:, 0]
                        y = points[:, 1]
                        num_curves = self.controller_object.number_of_std_curves
                        ellipse_scale = self.controller_object.number_of_std
                        for i in range(1, num_curves + 1):
                            ellipse_scale += ellipse_scale
                            w, h, theta = Helper.calculate_std(x, y, ellipse_scale=ellipse_scale)

                            ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                                          width=w, height=h,
                                          angle=theta, color=color)
                            ell.set_facecolor('none')
                            self.axes.add_artist(ell)

        if self.plot_type == '2d':
            x_lim = self.axes.get_xlim()
            y_lim = self.axes.get_ylim()
#AR: general_lim modified to be relative instead of absolute
            x = (x_lim[0] * (1+0.01*general_lim), x_lim[1] * (1+0.01*general_lim))
            y = (y_lim[0] * (1+0.01*general_lim), y_lim[1] * (1+0.01*general_lim))
            self.axes.set_xlim(x)
            self.axes.set_ylim(y)

        elif self.plot_type == '3d':
            x_lim = self.axes.get_xlim()
            y_lim = self.axes.get_ylim()
            z_lim = self.axes.get_zlim()
#AR: general_lim modified to be relative instead of absolute
            x = (x_lim[0]  * (1+0.01*general_lim), x_lim[1]  * (1+0.01*general_lim))
            y = (y_lim[0]  * (1+0.01*general_lim), y_lim[1]  * (1+0.01*general_lim))
            z = (z_lim[0]  * (1+0.01*general_lim), z_lim[1]  * (1+0.01*general_lim))
            self.axes.set_xlim(x)
            self.axes.set_ylim(y)
            self.axes.set_zlim(z)

        components = self.controller_object.get_components()

        if  MINIMAL_VERSION==True or self.controller_object.x_axis_pcs_radio_btn.isChecked():
            self.axes.set_xlabel('PC{}'.format(components[0] + 1))
        elif self.controller_object.x_axis_processes_radio_btn.isChecked():
            self.axes.set_xlabel('{}'.format(
                self.controller_object.x_axis_combobox.currentText()))

        if  MINIMAL_VERSION==True or self.controller_object.y_axis_pcs_radio_btn.isChecked():
            self.axes.set_ylabel('PC{}'.format(components[1] + 1))
        elif self.controller_object.y_axis_processes_radio_btn.isChecked():
            self.axes.set_ylabel('{}'.format(
                self.controller_object.y_axis_combobox.currentText()))

        if  MINIMAL_VERSION==False and self.controller_object.z_axis_pcs_radio_btn.isChecked() \
                and self.z_axis_combobox.currentText() != '-':
            self.axes.set_zlabel('PC{}'.format(components[2] + 1))
        elif MINIMAL_VERSION==False and self.controller_object.z_axis_processes_radio_btn.isChecked():
            try:
                self.axes.set_zlabel('{}'.format(
                    self.controller_object.z_axis_combobox.currentText()))
            except AttributeError:
                pass

        # self.draw_annotation = False

        if self.controller_object.shape_space_plot_show_back_ground_annotation_images_check_box.isChecked() \
                and (MINIMAL_VERSION==True or (self.controller_object.x_axis_pcs_radio_btn.isChecked() and \
                self.controller_object.y_axis_pcs_radio_btn.isChecked())):
            self.create_annotation_image(self.axes)
        elif self.controller_object.shape_space_plot_show_back_ground_annotation_images_check_box.isChecked():
            self.create_ls_annotation_image(self.axes)
        try:
            mean_contour = self.shape_space_obj.get_shape_space_mean_shape()
            Helper.save_contours_csv(self.controller_object.temp_directory,
                                     mean_contour,
                                     "shape_space_mean_shape")
        except AttributeError:
            self.logger.error('can not reconstruct mean shape!', exc_info=True)

    def create_annotation_image(self, axes):
        self.logger.info('Start creating annotation images...')

        num_rows = self.controller_object.shape_space_plot_background_number_of_rows
        num_cols = self.controller_object.shape_space_plot_background_number_of_columns

        # number of rows and columns
        x_lim = axes.get_xlim()
        y_lim = axes.get_ylim()

        y = y_lim[1] - y_lim[0]
        x = x_lim[1] - x_lim[0]

        row = y / (2 * num_rows)
        col = x / (2 * num_cols)

        all_image_points = []
        row_pos = y_lim[0]
        col_pos = x_lim[0]
        for i in range(2 * num_rows):
            for j in range(2 * num_cols):
                if (i % 2 != 0) and (j % 2 != 0):
                    all_image_points.append(np.array([col_pos, row_pos]))
                col_pos += col
            row_pos += row
            col_pos = x_lim[0]

        # mean shape
          

        reconstructed_data = self.shape_space_obj.get_reconstruct_pointSD([0, 0])
        DetectContour.draw_reconstructed_data(
            reconstructed_data,
            "{}/mean_shape.png".format(self.controller_object.temp_directory))

        images_zoom = self.controller_object.shape_space_plot_background_image_zoom
        images_rotation = self.controller_object.shape_space_plot_background_image_rotation
        #AR: get max-dimension for all images
        #   initialize to 1e-10 to guard agains't divide by 0
        max_dim = 1e-10
        for np_point in all_image_points:
            reconstructed_data = self.shape_space_obj.get_reconstruct_pointSD(np_point)       
            max_x = np.amax(reconstructed_data[::2])
            min_x = np.amin(reconstructed_data[::2])
            max_y = np.amax(reconstructed_data[1::2])
            min_y = np.amin(reconstructed_data[1::2])
            if max_x-min_x > max_dim:
                max_dim = max_x-min_x
            if max_y-min_y > max_dim:
                max_dim = max_y-min_y

        for np_point in all_image_points:
            reconstructed_data = self.shape_space_obj.get_reconstruct_pointSD(np_point)
            # divide contour by a scalar to reduce the cost of producing
            #AR: To fix - rotation crashes when GPA is used, otherwise fine (need to convert contours)
            reconstructed_data = self.shape_space_obj.rotate_contour(reconstructed_data,images_rotation)
            reconstructed_data /= self.annotation_images_scale*max_dim/1000#1000 #/255

            arr_img = DetectContour.draw_reconstructed_data(
                reconstructed_data,
                "{}/img.png".format(self.controller_object.temp_directory))

            imagebox = OffsetImage(arr_img, zoom=images_zoom, cmap='gray')
            imagebox.image.axes = self.axes

            xy = (np_point[0], np_point[1])

            annotate_image = AnnotationBbox(
                imagebox, xy=xy,
                xycoords='data',
                pad=0.01,
                bboxprops=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.0),
            )
            annotate_image.zorder = 0

            axes.add_artist(annotate_image)


    def create_ls_annotation_image(self, axes):
        self.logger.info('Start creating annotation images...')

        num_rows = self.controller_object.shape_space_plot_background_number_of_rows
        num_cols = self.controller_object.shape_space_plot_background_number_of_columns

        # number of rows and columns
        x_lim = axes.get_xlim()
        y_lim = axes.get_ylim()

        y = y_lim[1] - y_lim[0]
        x = x_lim[1] - x_lim[0]

        row = y / (2 * num_rows)
        col = x / (2 * num_cols)

        all_image_points = []
        row_pos = y_lim[0]
        col_pos = x_lim[0]
        for i in range(2 * num_rows):
            for j in range(2 * num_cols):
                if (i % 2 != 0) and (j % 2 != 0):
                    all_image_points.append(np.array([col_pos, row_pos]))
                col_pos += col
            row_pos += row
            col_pos = x_lim[0]

        a = self.controller_object.get_all_aligned_contours()
        a_array = []
        for cnt in a:
            a_array.append(np.array(cnt['aligned_contour']).flatten())

        regression = linear_model.LinearRegression()
        regression.fit(self.coordinate_values,a_array)
        reconstructed_data =  regression.predict([0,0]) #self.shape_space_obj.get_reconstruct_pointSD([0, 0])
        DetectContour.draw_reconstructed_data(
            reconstructed_data,
            "{}/mean_shape.png".format(self.controller_object.temp_directory))

        images_zoom = self.controller_object.shape_space_plot_background_image_zoom
        images_rotation = self.controller_object.shape_space_plot_background_image_rotation
        #AR: get max-dimension for all images
        #   initialize to 1e-10 to guard agains't divide by 0
        max_dim = 1e-10
        for np_point in all_image_points:
            reconstructed_data = regression.predict(np_point)[0]
            max_x = np.amax(reconstructed_data[::2])
            min_x = np.amin(reconstructed_data[::2])
            max_y = np.amax(reconstructed_data[1::2])
            min_y = np.amin(reconstructed_data[1::2])
            if max_x-min_x > max_dim:
                max_dim = max_x-min_x
            if max_y-min_y > max_dim:
                max_dim = max_y-min_y

        for np_point in all_image_points:
            reconstructed_data = regression.predict(np_point) 
            reconstructed_data = self.shape_space_obj.rotate_contour(reconstructed_data,images_rotation)
            reconstructed_data /= self.annotation_images_scale*max_dim/350 #/255

            arr_img = DetectContour.draw_reconstructed_data(
                reconstructed_data,
                "{}/img.png".format(self.controller_object.temp_directory))

            imagebox = OffsetImage(arr_img, zoom=images_zoom, cmap='gray')
            imagebox.image.axes = self.axes

            xy = (np_point[0], np_point[1])

            annotate_image = AnnotationBbox(
                imagebox, xy=xy,
                xycoords='data',
                pad=0.01,
                bboxprops=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.0),
            )
            annotate_image.zorder = 0

            axes.add_artist(annotate_image)

    def init_tweaks(self, axis=None):
        # check if we already initialized tweaks
        if self.method in ['LDA', 'LDA on Fourier coefficient']:
            self.controller_object.total_variance_lbl = 0.00
            total_number_of_components = self.shape_space_obj.get_total_number_of_lda()
        else:
            self.controller_object.total_variance_lbl = self.shape_space_obj.get_total_variance()
            total_number_of_components = self.shape_space_obj.get_total_number_of_pca()

        num_dimensions = self.shape_space_obj.get_total_number_of_pca()

        pervious_x_index = self.x_axis_combobox.currentIndex()
        pervious_y_index = self.y_axis_combobox.currentIndex()
        if MINIMAL_VERSION == False:
            pervious_z_index = self.z_axis_combobox.currentIndex()
        if pervious_x_index == pervious_y_index == 0:
            pervious_y_index = pervious_x_index + 1
            pervious_z_index = pervious_y_index + 1
        x_pc_button = True
        y_pc_button = True
        if MINIMAL_VERSION == False:
            z_pc_button = True
        if MINIMAL_VERSION == False:
            x_pc_button = self.controller_object.x_axis_pcs_radio_btn.isChecked()
            y_pc_button = self.controller_object.y_axis_pcs_radio_btn.isChecked()
            z_pc_button = self.controller_object.z_axis_pcs_radio_btn.isChecked()

        if self.x_axis_combobox.count() - 1 != total_number_of_components:
            if  x_pc_button and \
                            'PC 1' not in self.x_axis_combobox.currentText():
                if 1 < self.x_axis_combobox.count():
                    self.x_axis_combobox.clear()
                if self.x_axis_combobox.itemText(0) == '-':
                    self.x_axis_combobox.removeItem(0)
                for i in range(num_dimensions):
                    self.x_axis_combobox.addItem("PC {}".format(i + 1))

                if self.x_axis_combobox.findText("-") == -1:
                    self.x_axis_combobox.addItem("-")
                self.x_axis_combobox.setCurrentIndex(pervious_x_index)

        if self.y_axis_combobox.count() - 1 != total_number_of_components:
            if y_pc_button and \
                            'PC 1' not in self.y_axis_combobox.currentText():
                if 1 < self.y_axis_combobox.count():
                    self.y_axis_combobox.clear()
                if self.y_axis_combobox.itemText(0) == '-':
                    self.y_axis_combobox.removeItem(0)
                for i in range(num_dimensions):
                    self.y_axis_combobox.addItem("PC {}".format(i + 1))
                if self.y_axis_combobox.findText("-") == -1:
                    self.y_axis_combobox.addItem("-")
                self.y_axis_combobox.setCurrentIndex(pervious_y_index)

        if MINIMAL_VERSION == False and self.z_axis_combobox.count() - 1 != total_number_of_components:
            if z_pc_button and \
                            'PC 1' not in self.z_axis_combobox.currentText():
                if 1 < self.z_axis_combobox.count():
                    self.z_axis_combobox.clear()
                if self.z_axis_combobox.itemText(0) == '-':
                    self.z_axis_combobox.removeItem(0)
                if num_dimensions >= 3:
                    for i in range(num_dimensions):
                        self.z_axis_combobox.addItem("PC {}".format(i + 1))
                if self.z_axis_combobox.findText("-") == -1:
                    self.z_axis_combobox.addItem("-")
                self.z_axis_combobox.setCurrentIndex(
                    self.z_axis_combobox.findText("-"))

    def on_click_matplot(self, event):
        self.logger.info('Clicked on the shape space plot!')

        if len(self.all_artists) > 0:
            try:
                for data in self.all_artists:
                    data.remove()
                self.all_artists[:] = []
            except ValueError:
                self.all_artists[:] = []
                # self.logger.error('Error on removing annotate image!', exc_info=True)
                pass
                # self.figure.canvas.draw()
        try:
            point_str = self.axes.format_coord(event.xdata, event.ydata)
        except:
            return
        if self.plot_type == "3d":
            try:
                x, y, z = [float(l) for t in point_str.split(" ") if t for l in t.split("=")
                           if l != 'x' and l != 'y' and l != 'z' and l != ","]
            except ValueError:
                # print("Point NOT FOUND!", point_str)
                self.figure.canvas.draw()
                return

            np_point = np.array([x, y, z])
            x2, y2, _ = proj3d.proj_transform(x, y, z, self.axes.get_proj())
            np_point_2d = np.array([x2, y2])
            return
        else:
            try:
                x, y = [float(l) for t in point_str.split(" ") if t for l in t.split("=")
                        if l != 'x' and l != 'y']
            except ValueError:
                # print("Point NOT FOUND!", point_str)
                self.figure.canvas.draw()
                return
            x2 = x
            y2 = y
            np_point_2d = np.array([x2, y2])

        # === Add position of the clicked point
        # if x2 >= 0 and y2 >= 0:
        #     label_box_pos = (10, 40)
        # elif x2 <= 0 <= y2:
        #     label_box_pos = (0, 40)
        # elif x2 >= 0 >= y2:
        #     label_box_pos = (10, -50)
        # elif x2 <= 0 and y2 <= 0:
        #     label_box_pos = (0, -50)
        #
        # self.annotate_label = plt.annotate(
        #     "x: {} \n y: {}".format(x, y),
        #     xy=(x2, y2), xycoords='data',xytext=label_box_pos,
        #     textcoords='offset points', #ha='right', va='bottom',
        #     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        #     arrowprops=dict(arrowstyle='->',
        #                     connectionstyle="angle,angleA=0,angleB=-90,rad=3"))
        #
        # self.annotate_label.update_positions(self.figure.canvas.renderer)
        # self.all_artists.append(self.annotate_label)
        # =========================================

        # self.create_annotation_image(self.axes)
        if MINIMAL_VERSION == False:
            annotation_images_zoom = self.controller_object.shape_space_plot_annotation_images_zoom

        # ===== Reconstructed and show image of the clicked coordinate
        if MINIMAL_VERSION == False and self.controller_object.shape_space_plot_show_only_reconstructed_image_check_box.isChecked():
            try:
                reconstructed_data = self.shape_space_obj.get_reconstruct_pointSD(np_point_2d)
                reconstructed_data /= self.annotation_images_scale

                arr_img = DetectContour.draw_reconstructed_data(
                    reconstructed_data,
                    "{}/img.png".format(self.controller_object.temp_directory))

                imagebox = OffsetImage(arr_img, zoom=annotation_images_zoom, cmap='gray')
                imagebox.image.axes = self.axes

                if x2 >= 0 and y2 >= 0:
                    image_box_pos = (-80., -50.)
                elif x2 <= 0 <= y2:
                    image_box_pos = (80., -50.)
                elif x2 >= 0 >= y2:
                    image_box_pos = (-80., 50.)
                elif x2 <= 0 and y2 <= 0:
                    image_box_pos = (80., 50.)

                annotate_image = AnnotationBbox(imagebox, xy=(x2, y2),
                                                xybox=image_box_pos,
                                                xycoords='data',
                                                boxcoords="offset points",
                                                pad=0.5,
                                                bboxprops=dict(boxstyle="round", fc="w", ec="0.5",
                                                               alpha=0.1),
                                                arrowprops=dict(
                                                    arrowstyle="->",
                                                    connectionstyle="angle,angleA=0,angleB=90,rad=3")
                                                )

                self.axes.add_artist(annotate_image)
                self.all_artists.append(annotate_image)
            except:
                self.logger.error('Cannot reconstruct the image!', exc_info=True)
        # ================================================

        # ============ Show corresponding image of the clicked point
        # if self.controller_object.shape_space_plot_show_only_closest_original_contour_image_check_box.isChecked():
        _, _, result_image_path = self.get_image_name_from_point(np_point_2d)
        if result_image_path is not None:
            self.controller_object.display_image_in_shape_space_graphic_view(result_image_path)
        # ================================================

        self.figure.canvas.draw()
        self.logger.info('Drawing annotate items finished')

    def on_hover_matplotlib(self, event):

        try:
            point_str = self.axes.format_coord(event.xdata, event.ydata)
        except:
            return

        if self.plot_type == "3d":
            try:
                x, y, z = [float(l) for t in point_str.split(" ") if t for l in t.split("=")
                           if l != 'x' and l != 'y' and l != 'z' and l != ","]
            except ValueError:
                return
            # print("x=",x, "y=",y, "z=",z)
            np_point = np.array([x, y, z])
            x2, y2, _ = proj3d.proj_transform(x, y, z, self.axes.get_proj())
        else:
            try:
                x, y = [float(l) for t in point_str.split(" ") if t for l in t.split("=")
                        if l != 'x' and l != 'y']
            except ValueError:
                # print("Point NOT FOUND!", point_str)
                return

        text = "x: {0:.3f} \n y: {1:.3f}".format(x, y)

    def prepare_color_for_shape_space(self, reset_class_colors=False):
        self.logger.info('Preparing colors for shape space...')
        if self.controller_object.class_values != \
                self.shape_space_obj.get_all_class_values():
            self.controller_object.class_values = \
                self.shape_space_obj.get_all_class_values()
            self.controller_object.shape_class_colors_dict = \
                Helper.get_random_rgb_class_color(
                    self.controller_object.class_values)

        if self.controller_object.shape_class_colors_dict is None:
            self.controller_object.shape_class_colors_dict = \
                Helper.get_random_rgb_class_color(
                    self.controller_object.class_values)

        if reset_class_colors:
            self.controller_object.class_values = \
                self.shape_space_obj.get_all_class_values()

            self.controller_object.shape_class_colors_dict = \
                Helper.get_random_rgb_class_color(
                    self.controller_object.class_values)

        self.shape_space_obj.set_class_color_dict(
            self.controller_object.shape_class_colors_dict)

        self.logger.info('Shape space colors are prepared.')

    def export_plot(self, dst_directory, filename, plot_format):
        # remove extension
        filename = filename.split('.')[0]
        plot_filename = Helper.build_path(dst_directory, filename + '_Shape_Space.{}'.format(plot_format))
        self.figure.savefig(plot_filename, format=plot_format)

    def get_image_name_from_point(self, point):
        point = np.array(point)
        if point.shape[0] == 2:
            self.coordinate_values = self.coordinate_values[:, [0, 1]]

        closest_point, ind = Helper.get_closest_point(self.coordinate_values, point)
        aligned_contours_info_dict = self.controller_object.get_all_aligned_contours()

        image_name = aligned_contours_info_dict[ind[0]]['image_name']
        result_image_path = aligned_contours_info_dict[ind[0]]['image_path']
        image_number = None
        for k, v in self.controller_object.dict_of_images.items():
            if v == image_name:
                image_number = k

        if image_number is None:
            return None, None, None

        if MINIMAL_VERSION == True or not self.controller_object.always_show_closest_image_check_box.isChecked():
            min_distance = 0.015
            # normalize points to find distance
            x_l = self.axes.get_xlim()
            y_l = self.axes.get_ylim()
            point1 = [0, 0]
            point2 = [0, 0]
            point1[0] = (point[0] - x_l[0]) / (x_l[1] - x_l[0])
            point1[1] = (point[1] - y_l[0]) / (y_l[1] - y_l[0])
            point2[0] = (closest_point[0][0] - x_l[0]) / (x_l[1] - x_l[0])
            point2[1] = (closest_point[0][1] - y_l[0]) / (y_l[1] - y_l[0])

            dist = Helper.euclidean_distance(point1, point2)

            if dist > min_distance:
                return None, None, None

        status_bar_text = "{} - {}".format(image_number, image_name)

        self.controller_object.shape_space_image_lbl.setText(status_bar_text)
        # self.controller_object.update_status_bar(status_bar_text)
        return image_number, image_name, result_image_path

    @pyqtSlot()
    def work(self):
        self.sig_done.emit(1, "done")
