import numpy as np
import math
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QThread)
from scipy.linalg import orthogonal_procrustes

from .helper import Helper


class ProcrustesAnalysisWorkerThread(QThread):
    sig_done = pyqtSignal(int, str)

    def __init__(self, _controller_obj, _temp_directory, _resampled_contours_info_dict_list,
                 _mapped_main_cnt, _selected_method, _minimize_variance, parent=None):
        super(ProcrustesAnalysisWorkerThread, self).__init__(parent)
        self.controller_obj = _controller_obj
        self.temp_directory = _temp_directory
        self.image_contour = None
        self.scaling = True
        self.resampled_contours_info_dict_list = _resampled_contours_info_dict_list
        self.mapped_main_cnt = _mapped_main_cnt
        self.selected_option = _selected_method
        self.minimize_variance = _minimize_variance

        self.main_cnt = None
        self.other_cnts = None
        self.procrustes_result_dict_list = None
        self.generalized_proc_mean_shape_dict = None
        self.min_variance = None
        self.flipped_contours_info_dict = {}

    @pyqtSlot()
    def work(self):
        sel_method = self.selected_option.text(0)
#        sel_method = 'Align landmarks with Y-axis (2 landmarks)'
        if not self.resampled_contours_info_dict_list:
            self.sig_done.emit(0, "Can not find resampled contours!")
            return

        if sel_method == 'Simple Procrustes Analysis':
            self.main_cnt, self.other_cnts = self.procrustes_analysis(
                self.mapped_main_cnt,
                self.resampled_contours_info_dict_list)

            if self.minimize_variance:
                self.min_variance = self.calculate_variance(self.main_cnt, self.other_cnts)

                self.flip_contours_based_on_vriance()

        elif sel_method == 'Simple Procrustes Analysis (Without scaling)':

            self.scaling = False
            self.main_cnt, self.other_cnts = self.procrustes_analysis(
                self.mapped_main_cnt,
                self.resampled_contours_info_dict_list)

            if self.minimize_variance:
                self.min_variance = self.calculate_variance(self.main_cnt, self.other_cnts)

                self.flip_contours_based_on_vriance()

        elif sel_method == 'Generalized Procrustes Analysis':
            try:
                a, b = self.generalized_procrustes_analysis(
                    self.resampled_contours_info_dict_list)

                self.procrustes_result_dict_list = a
                self.generalized_proc_mean_shape_dict = b

                if self.minimize_variance:
                    self.min_variance = self.calculate_variance_gpa(a, b)

                    self.flip_contours_based_on_vriance_gpa()


            except:
                self.sig_done.emit(0, "Can not perform the operation!\n"
                                      "Please make sure that more than\n"
                                      "one sample, loaded and re-sampled!")
                return

        elif sel_method == 'Generalized Procrustes Analysis (Without scaling)':

            self.scaling = False
            try:
                a, b = self.generalized_procrustes_analysis(
                    self.resampled_contours_info_dict_list)

                if self.minimize_variance:
                    self.min_variance = self.calculate_variance_gpa(a, b)

                    self.flip_contours_based_on_vriance_gpa()
                self.procrustes_result_dict_list = a
                self.generalized_proc_mean_shape_dict = b
            except:
                Helper.critical_message("Can not perform the operation!\n"
                                        "Please make sure that more than\n"
                                        "one sample, loaded and re-sampled!")
                return

        elif sel_method == 'Align main-axis (2 Point)':
            try:
                if self.minimize_variance:
                    self.sig_done.emit(0, "Not supported for 2 point alignment!")
                self.scaling = False
                result_list, mean_shape_dict = self.yAlign(self.resampled_contours_info_dict_list)
                self.other_cnts = result_list
                self.main_cnt = mean_shape_dict
            except:
                self.sig_done.emit(0,"Can not perform the operation!\n"
                                        "Please make sure that more than\n"
                                         "one sample, loaded and re-sampled!")
                return
        else:
            self.sig_done.emit(0,'Click on sub-item')
            return

        if len(result_list)>0:
            self.sig_done.emit(1, 'done')
        else:
            self.sig_done.emit(0, "Can not perform the operation!\n"
                                        "Please make sure that more than\n"
                                        "one sample, loaded and re-sampled!")

    def calculate_variance(self, main_cnt, other_cnts):
        main_cnt -= np.mean(main_cnt, 0)
        main_cnt = np.vstack(main_cnt).squeeze().flatten()
        all_points = main_cnt

        for data in other_cnts:
            mapped_resampled_cnt = data['resampled_cnt']

            if mapped_resampled_cnt:
                cnt = np.vstack(mapped_resampled_cnt).squeeze()
                try:
                    cnt = np.delete(cnt, 2, axis=1)
                except IndexError:
                    pass

                # translate to the origin
                cnt -= np.mean(cnt, 0)
                cnt = np.vstack(cnt).squeeze().flatten()
                if all_points.any():
                    all_points = np.c_[all_points, cnt]
                else:
                    all_points = cnt

        covariance = np.cov(all_points)
        variance = np.trace(covariance)

        return variance

    def flip_contours_based_on_vriance(self):
        counter = 0
        self.flipped_contours_info_dict = {}
        temp_flipped = {}  # {image_path : resample_points}

        while counter < len(self.resampled_contours_info_dict_list):
            self.new_resampled_contours_info_dict_list = []
            for i, data in enumerate(self.resampled_contours_info_dict_list):
                resampled_points = data['resampled_cnt']
                result_image_path = data['result_image_path']
                if result_image_path in temp_flipped.keys():
                    resampled_points = temp_flipped[result_image_path]['resampled_contour']

                landmarks = Helper.find_landmarks(resampled_points,
                                                  self.temp_directory,
                                                  result_image_path)[0]

                if i == counter:
                    flipping_result_image_path = result_image_path
                    flipping_texture_width = data['texture_width']
                    flipping_texture_height = data['texture_height']
                    flipping_color_rgb = data['color_rgb']
                    # remove z coordinate
                    resampled_points = Helper.change_number_of_contour_coordinates(resampled_points)
                    # flip the contour and its landmarks
                    resampled_points, landmarks = Helper.flip_contour_points(
                        [resampled_points], landmarks)

                    flipping_resampled_points = Helper.convert_listoflists_to_contour_format(
                        resampled_points, -1)
                    self.new_resampled_contours_info_dict_list.append({
                        'resampled_cnt': flipping_resampled_points,
                        'texture_width': data['texture_width'],
                        'texture_height': data['texture_height'],
                        'result_image_path': data['result_image_path'],
                        'color_rgb': data['color_rgb']
                    })

                else:
                    self.new_resampled_contours_info_dict_list.append({
                        'resampled_cnt': resampled_points,
                        'texture_width': data['texture_width'],
                        'texture_height': data['texture_height'],
                        'result_image_path': data['result_image_path'],
                        'color_rgb': data['color_rgb']
                    })

            main_cnt, other_cnts = self.procrustes_analysis(
                self.mapped_main_cnt,
                self.new_resampled_contours_info_dict_list)

            variance = self.calculate_variance(main_cnt, other_cnts)

            # print("current variance=", variance)

            if self.min_variance > variance:
                # print("flipped = ", flipping_result_image_path)

                # we need to make a dict like the one we send to prepare_resampled_contours
                # before performing procrustes analysis
                self.flipped_contours_info_dict[flipping_result_image_path] = {
                    'resampled_contour': flipping_resampled_points,
                    'texture_width': flipping_texture_width,
                    'texture_height': flipping_texture_height,
                    'image_path': flipping_result_image_path,
                    'color_rgb': flipping_color_rgb
                }
                # Here we keep the mapped version
                temp_flipped[flipping_result_image_path] = {
                    'mapped_resampled_contour': flipping_resampled_points
                }

                # Helper.save_resampled_csv(self.temp_directory,
                #                           contour, flipping_result_image_path)
                self.min_variance = variance

            counter += 1

    def calculate_variance_gpa(self, result_dict_list, meanshape_dict_list):
        all_points = np.array([])
        for data in result_dict_list:
            cnt = np.vstack(data['resampled_cnt']).squeeze()
            cnt = Helper.change_number_of_contour_coordinates(cnt)
            cnt = np.vstack(cnt).squeeze().flatten()

            if all_points.any():
                all_points = np.c_[all_points, cnt]
            else:
                all_points = cnt

        mean_cnt = np.vstack(meanshape_dict_list['mean_shape_cnt']).squeeze()
        mean_cnt = np.delete(mean_cnt, 2, axis=1)
        mean_cnt = np.vstack(mean_cnt).squeeze().flatten().reshape(-1, 1)
        all_points -= mean_cnt

        covariance = np.cov(all_points)
        variance = np.trace(covariance)

        # from sklearn.decomposition.pca import PCA
        # x = all_points.T
        # self.pca = PCA()
        # self.pca.fit(x)
        #
        # self.X_transformed = self.pca.transform(x)
        # variance = sum(self.pca.explained_variance_)

        return variance

    def flip_contours_based_on_vriance_gpa(self):
        counter = 0
        self.flipped_contours_info_dict = {}
        temp_flipped = {}  # {image_path : resample_points}

        while counter < len(self.resampled_contours_info_dict_list):
            self.new_resampled_contours_info_dict_list = []
            for i, data in enumerate(self.resampled_contours_info_dict_list):
                resampled_points = data['resampled_cnt']
                result_image_path = data['result_image_path']
                if result_image_path in temp_flipped.keys():
                    resampled_points = temp_flipped[result_image_path]['mapped_resampled_contour']

                    # resampled_points = Helper.map_from_image_to_opengl(contour, [0, 0, 0, 0],
                    #         data['texture_width'], data['texture_height'])
                    # resampled_points = [resampled_points[0].reshape(-1, 1, 3)]

                # meta_data = Helper.read_metadata_from_csv(self.temp_directory,
                #                                           result_image_path)

                landmarks = Helper.find_landmarks(resampled_points,
                                                  self.temp_directory,
                                                  result_image_path)[0]

                if i == counter:
                    # print('i=',i)
                    flipping_result_image_path = result_image_path
                    flipping_texture_width = data['texture_width']
                    flipping_texture_height = data['texture_height']
                    flipping_color_rgb = data['color_rgb']
                    # remove z coordinate
                    resampled_points = Helper.change_number_of_contour_coordinates(resampled_points)
                    # flip the contour and its landmarks
                    resampled_points, landmarks = Helper.flip_contour_points(
                        [resampled_points], landmarks)
                    # add z coordinate
                    # z = -1 * np.ones((resampled_points[0].shape[0], 1))
                    # temp = np.vstack(resampled_points[0]).squeeze()
                    #
                    # resampled_points = np.c_[temp, z]
                    # resampled_points = [np.array(resampled_points.reshape(-1, 1, 3))]
                    flipping_resampled_points = Helper.convert_listoflists_to_contour_format(
                        resampled_points, -1)
                    self.new_resampled_contours_info_dict_list.append({
                        'resampled_cnt': flipping_resampled_points,
                        'texture_width': data['texture_width'],
                        'texture_height': data['texture_height'],
                        'result_image_path': data['result_image_path'],
                        'color_rgb': data['color_rgb']
                    })

                else:
                    self.new_resampled_contours_info_dict_list.append({
                        'resampled_cnt': resampled_points,
                        'texture_width': data['texture_width'],
                        'texture_height': data['texture_height'],
                        'result_image_path': data['result_image_path'],
                        'color_rgb': data['color_rgb']
                    })

            a, b = self.generalized_procrustes_analysis(
                self.new_resampled_contours_info_dict_list)

            variance = self.calculate_variance_gpa(a, b)

            # print("current variance=", variance)

            if self.min_variance > variance:
                # print("flipped = ", flipping_result_image_path)
                # meta_data['landmarks'] = landmarks
                # Helper.save_metadata_to_csv(self.temp_directory,
                #                             flipping_result_image_path, meta_data)

                # mapped_back_to_image = Helper.map_contour_from_opengl_to_image2(
                #     flipping_resampled_points, flipping_texture_width,
                #     flipping_texture_height, [0, 0, 0, 0])

                # we need to make a dict like the one we send to prepare_resampled_contours
                # before performing procrustes analysis
                self.flipped_contours_info_dict[flipping_result_image_path] = {
                    'resampled_contour': flipping_resampled_points,
                    'texture_width': flipping_texture_width,
                    'texture_height': flipping_texture_height,
                    'image_path': flipping_result_image_path,
                    'color_rgb': flipping_color_rgb
                }
                # Here we keep the mapped version
                temp_flipped[flipping_result_image_path] = {
                    'resampled_contour': flipping_resampled_points
                }

                # Helper.save_resampled_csv(self.temp_directory,
                #                           contour, flipping_result_image_path)
                self.min_variance = variance

            counter += 1

    @staticmethod
    def compute_centroid(cnt_points, dtype='float32'):
        # m = cv2.moments(cnt_points[0])
        # cx = m['m10'] / m['m00']
        # cy = m['m01'] / m['m00']
        cnt_squeezed = np.vstack(cnt_points[0]).squeeze()
        x = [p[0] for p in cnt_squeezed]
        y = [p[1] for p in cnt_squeezed]

        cx = sum(x) / len(cnt_squeezed)
        cy = sum(y) / len(cnt_squeezed)

        return [np.array([[[cx, cy]]], dtype=dtype)]

    @staticmethod
    def translate_to_center(cnt_points, cnt_center):
        # print("cnt_center=", cnt_center)
        # print("cnt= ", cnt_points)
        # translate_matrix = pyrr.matrix44.create_from_translation(
        #     [cnt_center[0], cnt_center[1], 0, 0])

        dist_x = 0 - cnt_center[0][0]
        dist_y = 0 - cnt_center[0][1]
#        print("dist_x=", dist_x, "dist_y=", dist_y)

        cnt_points[0][:, :, 0] += dist_x
        cnt_points[0][:, :, 1] += dist_y
        translated_cnt = cnt_points
#        print("translated_cnt=", translated_cnt)
        cnt_points = np.vstack(cnt_points).squeeze()
        mtx1 = np.delete(cnt_points, 2, axis=1)
        mtx1 -= np.mean(mtx1, 0)
        translated_cnt = mtx1
#        print("translated_cnt=", translated_cnt)
        return translated_cnt

    # def convert_listoflists_to_contour_format(self, input_list, add_to_z=None):
    #     if add_to_z:
    #         contour = np.c_[input_list, add_to_z * np.ones(np.asarray(
    #             input_list).shape[0])].reshape((-1, 1, 3))
    #     else:
    #         contour = np.array(input_list).reshape((-1, 1, 2))
    #
    #     contour = [np.asarray(contour)]
    #
    #     return contour

    # def find_rotation_degree(self, first_cnt, second_cnt):
    #     # first_cnt_landmarks = DetectContour.find_landmarks(first_cnt,
    #     #                                                    dtype='float32')
    #     # second_cnt_landmarks = DetectContour.find_landmarks(second_cnt,
    #     #                                                     dtype='float32')
    #     # print(first_cnt_landmarks)
    #     first_cnt = self.convert_listoflists_to_contour_format(first_cnt)
    #     second_cnt = self.convert_listoflists_to_contour_format(second_cnt)
    #     center_first = self.compute_centroid(first_cnt, dtype='float32')
    #     # print("center first= ", center_first[0])
    #     self.translate_to_center(first_cnt, center_first[0])
    #
    #     #print(cv2.minAreaRect(first_cnt[0]))
    #     # print(second_cnt_landmarks)
    #     center_second = self.compute_centroid(second_cnt, dtype='float32')
    #     print("center second= ", center_second[0])
    #
    #     self.translate_to_center(second_cnt, center_second[0])
    #
    #     first_cnt_landmarks = DetectContour.find_landmarks(first_cnt,
    #                                                        dtype='float32')
    #     second_cnt_landmarks = DetectContour.find_landmarks(second_cnt,
    #                                                         dtype='float32')
    #     print("land_marks = ", first_cnt_landmarks, '\n',  second_cnt_landmarks)
    #
    #     #r = np.dot(second_cnt_landmarks, np.linalg.inv(first_cnt_landmarks))
    #

    def procrustes_analysis(self, main_cnt, contours_dict_list):

        main_contour = np.vstack(main_cnt).squeeze()
        try:
            main_contour = np.delete(main_contour, 2, axis=1)
        except IndexError:
            pass

        # print("main_contour=", main_contour)
        max_texture_width, max_texture_height = (0, 0)
        result_list = []

        # translate to the origin
        main_contour -= np.mean(main_contour, 0)
        main_cnt_s = ProcrustesAnalysisWorkerThread.calculate_scaling(main_contour)
        main_contour /= main_cnt_s
        self.mu = main_cnt_s
        for data in contours_dict_list:
            mapped_resampled_cnt = data['resampled_cnt']
            color_rgb = data['color_rgb']

            if mapped_resampled_cnt:
                contour = np.vstack(mapped_resampled_cnt).squeeze()
                try:
                    contour = np.delete(contour, 2, axis=1)
                except IndexError:
                    pass
            else:
                continue
            try:
#                print(contour)
                # translate to the origin
                contour -= np.mean(contour, 0)

                if not np.array_equal(contour, main_contour):
                    s = ProcrustesAnalysisWorkerThread.calculate_scaling(contour)

                    R, _, disparity = \
                        self.procrustes(main_contour, contour)
#                    print(R)
                    if self.scaling:
                        contour /= s
                        new_contour = np.dot(contour, R.T)
                    else:
                        new_contour = np.dot(contour, R.T)

                    new_contour = Helper.convert_listoflists_to_contour_format(
                        new_contour, -1)

                    cnt_center = self.compute_centroid(new_contour)

                    if 1 - abs(R[0, 1]) < 0.1 and 1 - abs(R[1, 0]) < 0.1:
                        texture_height = data['texture_width']
                        texture_width = data['texture_height']
                    else:
                        texture_width = data['texture_width']
                        texture_height = data['texture_height']
                    # keep the maximums
                    if texture_width > max_texture_width and texture_height > max_texture_height:
                        max_texture_width = texture_width
                        max_texture_height = texture_height
                    # if texture_height > max_texture_height:
                    #     max_texture_height = texture_height

                    result_list.append({'resampled_cnt': new_contour,
                                        'texture_width': texture_width,
                                        'texture_height': texture_height,
                                        'mapped_cnt_center': cnt_center,
                                        'result_image_path':
                                            data['result_image_path'],
                                        'color_rgb': color_rgb})
                    # else:
                    #     print("Hi")
                    #     result_list.append({'mapped_cnt': main_contour,
                    #                         'texture_width': max_texture_width,
                    #                         'texture_height': max_texture_height,
                    #                         'mapped_cnt_center': self.compute_centroid(main_contour),
                    #                         'result_image_path':
                    #                             data['result_image_path'],
                    #                         'color_rgb': color_rgb})
            except ValueError as e:
                print(e)

        for data in result_list:
            data['texture_width'] = max_texture_width
            data['texture_height'] = max_texture_height

        main_contour = Helper.convert_listoflists_to_contour_format(
            main_contour, -1)

        return main_contour, result_list

    def procrustes(self, data1, data2):
        r"""Procrustes analysis, a similarity test for two data sets.
        Each input matrix is a set of points or vectors (the rows of the matrix).
        The dimension of the space is the number of columns of each matrix. Given
        two identically sized matrices, procrustes standardizes both such that:
        - :math:`tr(AA^{T}) = 1`.
        - Both sets of points are centered around the origin.
        Procrustes ([1]_, [2]_) then applies the optimal transform to the second
        matrix (including scaling/dilation, rotations, and reflections) to minimize
        :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
        pointwise differences between the two input datasets.
        This function was not designed to handle datasets with different numbers of
        datapoints (rows).  If two data sets have different dimensionality
        (different number of columns), simply add columns of zeros to the smaller
        of the two.
        Parameters
        ----------
        data1 : array_like
            Matrix, n rows represent points in k (columns) space `data1` is the
            reference data, after it is standardised, the data from `data2` will be
            transformed to fit the pattern in `data1` (must have >1 unique points).
        data2 : array_like
            n rows of data in k space to be fit to `data1`.  Must be the  same
            shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
        Returns
        -------
        mtx1 : array_like
            A standardized version of `data1`.
        mtx2 : array_like
            The orientation of `data2` that best fits `data1`. Centered, but not
            necessarily :math:`tr(AA^{T}) = 1`.
        disparity : float
            :math:`M^{2}` as defined above.
        Raises
        ------
        ValueError
            If the input arrays are not two-dimensional.
            If the shape of the input arrays is different.
            If the input arrays have zero columns or zero rows.
        See Also
        --------
        scipy.linalg.orthogonal_procrustes
        Notes
        -----
        - The disparity should not depend on the order of the input matrices, but
          the output matrices will, as only the first output matrix is guaranteed
          to be scaled such that :math:`tr(AA^{T}) = 1`.
        - Duplicate data points are generally ok, duplicating a data point will
          increase its effect on the procrustes fit.
        - The disparity scales as the number of points per input matrix.
        References
        ----------
        .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
        .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".
        Examples
        --------
        # >>> from scipy.spatial import procrustes
        # The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
        # ``a`` here:
        # >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
        # >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
        # >>> mtx1, mtx2, disparity = procrustes(a, b)
        # >>> round(disparity)
        0.0
        """
        mtx1 = np.array(data1, dtype=np.double, copy=True)
        mtx2 = np.array(data2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = orthogonal_procrustes(mtx1, mtx2)

        mtx2 = np.dot(mtx2, R.T) * s

        # measure the dissimilarity between the two datasets
        disparity = np.sum(np.square(mtx1 - mtx2))

        return R, s, disparity

    def compute_mean_shape(self, contours_dict_list):
        contours = []
        for data in contours_dict_list:
            mapped_cnt = data['resampled_cnt']
            contour = np.vstack(mapped_cnt).squeeze()
            try:
                contour = np.delete(contour, 2, axis=1)
            except IndexError:
                pass
            contours.append(contour)

        contours = np.array(contours)

        mean_shape = []
        for i in range(len(contours[0, :])):
            mean_shape.append(contours[:, i].mean(axis=0))

        return np.array(mean_shape, dtype=np.float32)

    def find_contours_point_by_point_distance(self, contour1, contour2):
        contour1 = np.vstack(contour1).squeeze()
        contour2 = np.vstack(contour2).squeeze()
        try:
            contour1 = Helper.change_number_of_contour_coordinates(contour1)
        except IndexError:
            pass

        try:
            contour2 = Helper.change_number_of_contour_coordinates(contour2)
        except IndexError:
            pass

        diff = contour1 - contour2
        diff_square = np.power(diff, 2)

        dist = np.sqrt(np.sum(diff_square, axis=1))

        return dist

    def yAlign(self, contours_info_dict_list):
        max_texture_width, max_texture_height = (0, 0)
        result_list = []
        unsampled_contours = False
        for i, data in enumerate(contours_info_dict_list):
#            print(i)
#            print(data)
            resampled_points = data['resampled_cnt']
            if len(resampled_points)==0:
                unsampled_contours = True
                break
            result_image_path = data['result_image_path']
            original_contour = Helper.load_contours_from_csv(self.temp_directory,
                                                            result_image_path)
            landmarks = Helper.find_landmarks(original_contour,
                                                  temp_dir=self.temp_directory,
                                                  image_path=result_image_path)[0]            

            avg_pos = (landmarks[1]+landmarks[0])*0.5
            p2 = landmarks[1]-avg_pos
            p1 = landmarks[0]-avg_pos
            theta = -(math.atan2(p1[0,1],p1[0,0]))-3.14159/2.0
            new_resampled_points = resampled_points-avg_pos
            R = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
            new_resampled_points = np.dot(new_resampled_points,R.T)

#            print("old/new transition")
#            avg_pos = (landmarks[1]+landmarks[0])*0.5
#            p1 = landmarks[1]-avg_pos
#            p2 = landmarks[0]-avg_pos
#            theta = -(math.atan2(p1[0,1],p1[0,0]))
#            if abs(theta)>1.0:
#                theta=3.14159
#            else:
#                theta=0
#            theta = theta + 3.14159/2.0
#            cnt_center = 0.5*(resampled_points[0][0]+resampled_points[0][60])
#            cnt_center[0,2]=0
 #           new_resampled_points = resampled_points-cnt_center
#            R = np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])
#            new_resampled_points = np.dot(new_resampled_points,R.T)
            color_rgb = data['color_rgb']
            contour = new_resampled_points #np.vstack(new_resampled_points).squeeze()
            try:
                contour = np.delete(contour, 2, axis=1)
            except IndexError:
                pass
            try:
                new_contour = Helper.convert_listoflists_to_contour_format(
                                contour, -1)
                
                cnt_center = self.compute_centroid(new_contour)
                
                if 1 - abs(R[0, 1]) < 0.1 and 1 - abs(R[1, 0]) < 0.1:
                    texture_height = data['texture_width']
                    texture_width = data['texture_height']
                else:
                    texture_width = data['texture_width']
                    texture_height = data['texture_height']
                            # keep the maximums
                if texture_width > max_texture_width and texture_height > max_texture_height:
                    max_texture_width = texture_width
                    max_texture_height = texture_height
                            # if texture_height > max_texture_height:
                            #     max_texture_height = texture_height
                result_list.append({'resampled_cnt': new_resampled_points,
                                                'mapped_resampled_cnt': new_contour,
                                                'texture_width': texture_width,
                                                'texture_height': texture_height,
                                                'mapped_cnt_center': cnt_center,
                                                'result_image_path':
                                                    data['result_image_path'],
                                                'color_rgb': color_rgb})
                
            except ValueError as e:
                print(e)

        mean_shape_dict = {}
        if len(result_list)>0 and unsampled_contours == False :
            for data in result_list:
                data['texture_width'] = max_texture_width
                data['texture_height'] = max_texture_height
            mean_shape_cnt = self.compute_mean_shape(result_list)                   
            mean_shape_cnt = np.vstack(mean_shape_cnt).squeeze()
            mean_shape_cnt = Helper.convert_listoflists_to_contour_format(
            mean_shape_cnt, -1)
            mean_scale_factor = 1
            mean_shape_center = self.compute_centroid(mean_shape_cnt)
            mean_shape_width = result_list[0]['texture_width']
            mean_shape_height = result_list[0]['texture_height']
        #
            mean_shape_dict = {'mean_shape_cnt': mean_shape_cnt,
                           'mean_shape_width': mean_shape_width,
                           'mean_shape_height': mean_shape_height,
                           'mean_shape_center': mean_shape_center,
                           'mean_shape_scale_factor': mean_scale_factor
                           }
        else:
            result_list = []
        return result_list, mean_shape_dict

    def generalized_procrustes_analysis(self, contours_dict_list):
        epsilon = 0.001
#        print("In generalized procrustes analysis")
        mean_shape = contours_dict_list[0]
        mean_shape_cnt = mean_shape['resampled_cnt']
        mean_shape_width = mean_shape['texture_width']
        mean_shape_height = mean_shape['texture_height']

        while True:
            _, result_list = self.procrustes_analysis(mean_shape_cnt,
                                                      contours_dict_list)
            new_mean_shape_cnt = self.compute_mean_shape(result_list)

            dist = self.find_contours_point_by_point_distance(
                new_mean_shape_cnt, mean_shape_cnt)

            diff = np.mean(dist)
            # print("diff=", diff)

            mean_shape_cnt = new_mean_shape_cnt

            if diff <= epsilon:
                break

        mean_shape_cnt = np.vstack(mean_shape_cnt).squeeze()
        mean_shape_cnt = Helper.convert_listoflists_to_contour_format(
            mean_shape_cnt, -1)

        # now we do the alignment with the calculated mean shape
        _, result_list = self.procrustes_analysis(mean_shape_cnt,
                                                  contours_dict_list)

        mean_shape_center = self.compute_centroid(mean_shape_cnt)
        mean_scale_factor = ProcrustesAnalysisWorkerThread.calculate_scaling(
            mean_shape_cnt)

        mean_shape_dict = {'mean_shape_cnt': mean_shape_cnt,
                           'mean_shape_width': mean_shape_width,
                           'mean_shape_height': mean_shape_height,
                           'mean_shape_center': mean_shape_center,
                           'mean_shape_scale_factor': mean_scale_factor
                           }
        variance = self.calculate_variance_gpa(result_list, mean_shape_dict)

        return result_list, mean_shape_dict

    @staticmethod
    def calculate_scaling(contour):
        points = np.vstack(contour).squeeze()
        k = points.shape[0]
        points_mean = points.mean(axis=0)
        points_trans = points - points_mean
        points_sum = np.square(points_trans).sum(axis=0)

        scale = np.sqrt(points_sum.sum() / k)

        return scale
