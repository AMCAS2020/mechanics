import traceback
import math
import numpy as np
from sklearn.decomposition.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .elliptical_fourier_descriptor import calculate_efd, inverse_transform, normalize_efd
from .helper import Helper


class ShapeSpaceCalculator:
    def __init__(self, contours_dict_list, _method, _n_component=None, _components=None, **kwargs):
        self._contours_dict_list = contours_dict_list
        self._method = _method
        self.prepared_data = np.array([])
        self.prepared_classes = np.array([])
        self.pca = None
        self.lda_clf = None

        self.n_compo = _n_component
        if _components:
            self.components = _components

        self.kwargs = kwargs

        self.mu = None
        self.X_transformed = None
        self.harmonic = None
        self.class_color = {}
        self.column_color_dict = {}
        self.class_values = []  # for elliptical fourier (because it uses coefficients and not contours)

        self._perform_pca_before_lda = False

    def get_class_color_dict(self):
        return self.class_color

    def get_all_class_values(self):
        class_values = []
        for cnt in self._contours_dict_list:
            if cnt['class_value'] not in class_values:
                class_values.append(cnt['class_value'])

        return class_values

    def set_class_color_dict(self, class_color_dict):
        self.class_color = class_color_dict

    def map_class_to_color_and_value(self):
        column = 0
        for cnt in self._contours_dict_list:
            self.column_color_dict[column] = \
                {"color": self.class_color[cnt['class_value']],
                 "class_value": cnt['class_value']}

            column += 1

    def prepare_vectors(self):
        class_values = []
        for cnt in self._contours_dict_list:
            c = np.array(cnt['aligned_contour']).flatten()

            if self.prepared_data.any():
                self.prepared_data = np.c_[self.prepared_data, c]
            else:
                self.prepared_data = c

            try:
                class_values.append(cnt['class_value'])
            except Exception:
                tb = traceback.format_exc()
                print(tb)
                continue

    def separate_points_by_class(self, coordinate_values=None):
        """
        this method separate the transformed points into their classes
        :return: dictionary ({class: []})
                The key is class_value and the value is a list of points
        """
        class_cnt = {}
        if coordinate_values is None:
            coordinate_values = self.get_shape_space_coordinate_values()

        for i in range(len(self.column_color_dict)):
            class_value = self.column_color_dict[i]['class_value']
            point = coordinate_values[i]

            try:
                if class_cnt[class_value]:
                    class_cnt[class_value].append(point)
            except KeyError as e:
                class_cnt[class_value] = [point]
                continue

        return class_cnt

    def get_class_color(self, class_value):
        """
        find the color of the given class
        :param class_value: int
                            value of the class
        :return: list
                list of RGB color
        """
        return self.class_color[class_value]

    def get_colors_of_data(self):
        color = []
        for i in range(len(self.column_color_dict)):
            color.append(self.column_color_dict[i]["color"])
        return color

    def get_shape_space_coordinate_values(self):
        if self.n_compo:
            return self.X_transformed[:, :self.n_compo]
        else:
            max_component = self.X_transformed.shape[1] - 1
            for i in range(len(self.components)):
                if self.components[i] > max_component:
                    self.components[i] = max_component

            return self.X_transformed[:, self.components]

#AR: Scale coordinates by SD to get normalized values
    def get_shape_space_coordinate_valuesSD(self):
        coord = self.get_shape_space_coordinate_values()
        sd = np.sqrt(self.get_variance());
        sd = 1/sd
        return sd*coord


    # ============================== PCA ==================
    def compute_pca(self, _n_components=None):
        x = self.prepared_data.T
        self.mu = np.mean(x, axis=0)
        # save mean of the data
        # np.savetxt("mean_prepared_data.csv", self.mu, delimiter=',', header='mean', comments="")

        self.pca = PCA(n_components=_n_components)
        self.pca.fit(x)

        self.X_transformed = self.pca.transform(x)

    def get_total_number_of_pca(self):
        return self.X_transformed.shape[1]

    def get_percentage_of_variance(self):
        if self._method in ['PCA', 'Elliptical Fourier Descriptors']:
            evr = self.pca.explained_variance_ratio_[self.components]
        elif self._method in ['LDA', 'LDA on Fourier coefficient']:
            print("ad=", self.lda_clf.explained_variance_ratio_, self.lda_clf.explained_variance_ratio_.sum())
            # pc1_ratio = self.lda_clf.explained_variance_ratio_[0] /
            evr = self.lda_clf.explained_variance_ratio_[self.components]
            if evr.sum() > 1.0:
                evr = [0.0 for _ in range(len(self.lda_clf.explained_variance_ratio_[self.components]))]
                print(evr)
        return evr

#AR: total variance, as opposed to percentage
    def get_variance(self):
        if self._method in ['PCA', 'Elliptical Fourier Descriptors']:
            evr = self.pca.explained_variance_[self.components]
        elif self._method in ['LDA', 'LDA on Fourier coefficient']:
            print("ad=", self.lda_clf.explained_variance_, self.lda_clf.explained_variance_.sum())
            # pc1_ratio = self.lda_clf.explained_variance_ratio_[0] /
            evr = self.lda_clf.explained_variance_[self.components]
            if evr.sum() > 1.0:
                evr = [0.0 for _ in range(len(self.lda_clf.explained_variance_[self.components]))]
                print(evr)
        return evr

    def get_reconstruct_point(self, point):
        if self._method == "PCA":
            return self.get_reconstruct_pca(point)

        elif self._method == "Elliptical Fourier Descriptors":
            return self.get_reconstructed_efd(point)

        elif self._method == 'LDA':
            return self.get_reconstructed_lda(point)

        elif self._method == 'LDA on Fourier coefficient':
            return self.get_reconstructed_lda_efd(point)

#AR: reconstruct based on SD normalized
    def get_reconstruct_pointSD(self, point):
        sd = np.sqrt(self.get_variance());
        npoint = sd*point
        return self.get_reconstruct_point(npoint)

    def get_reconstruct_pca(self, point):
        eigenvector = self.pca.components_[[0, 1], :]

        x_hat = np.dot(point,
                       eigenvector)
        x_hat += self.mu

        return x_hat

    def rotate_contour(self,contour,theta):
        R = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        return np.dot(contour,R.T)

    # ============================== elliptical_fourier_descriptor ========
#Adaptation of pyefd normalize_efd code to allow for only scale invariance
#rotates to allow for scale invariance and then rotates back
    def normalize_size_only_efd(self,coeffs):
        """Normalizes an array of Fourier coefficients.
        See [#a]_ and [#b]_ for details.
        :param numpy.ndarray coeffs: A ``[n x 4]`` Fourier coefficient array.
        :param bool size_invariant: If size invariance normalizing should be done as well.
            Default is ``True``.
        :return: The normalized ``[n x 4]`` Fourier coefficient array.
        :rtype: :py:class:`numpy.ndarray`
        """
        try:
            _range = xrange
        except NameError:
            _range = range
        # Make the coefficients have a zero phase shift from
        # the first major axis. Theta_1 is that shift angle.
        theta_1 = 0.5 * np.arctan2(
            2 * ((coeffs[0, 0] * coeffs[0, 1]) + (coeffs[0, 2] * coeffs[0, 3])),
            (
                (coeffs[0, 0] ** 2)
                - (coeffs[0, 1] ** 2)
                + (coeffs[0, 2] ** 2)
                - (coeffs[0, 3] ** 2)
            ),
        )
        # Rotate all coefficients by theta_1.
        for n in _range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = np.dot(
                np.array(
                    [
                        [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                        [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                    ]
                ),
                np.array(
                    [
                        [np.cos(n * theta_1), -np.sin(n * theta_1)],
                        [np.sin(n * theta_1), np.cos(n * theta_1)],
                    ]
                ),
            ).flatten()

        # Make the coefficients rotation invariant by rotating so that
        # the semi-major axis is parallel to the x-axis.
        psi_1 = np.arctan2(coeffs[0, 2], coeffs[0, 0])
        psi_rotation_matrix = np.array(
            [[np.cos(psi_1), np.sin(psi_1)], [-np.sin(psi_1), np.cos(psi_1)]]
        )
        # Rotate all coefficients by -psi_1.
        for n in _range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = psi_rotation_matrix.dot(
                np.array(
                    [
                        [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                        [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                    ]
                )
            ).flatten()

 
            # Obtain size-invariance by normalizing.
            coeffs /= np.abs(coeffs[0, 0])
        #undo psi_rotation
        psi_rotation_matrix = np.array(
            [[np.cos(psi_1), -np.sin(psi_1)], [np.sin(psi_1), np.cos(psi_1)]]
        )
        # Rotate all coefficients by -psi_1.
        for n in _range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = psi_rotation_matrix.dot(
                np.array(
                    [
                        [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                        [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                    ]
                )
            ).flatten()

        #Undo theta_1 rotation
        for n in _range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = np.dot(
                np.array(
                    [
                        [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                        [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                    ]
                ),
                np.array(
                    [
                        [np.cos(n * theta_1), np.sin(n * theta_1)],
                        [-np.sin(n * theta_1), np.cos(n * theta_1)],
                    ]
                ),
            ).flatten()

        return coeffs

    def compute_efd(self, harmonic):
        self.column_color_dict = {}
        self.harmonic = harmonic
        self.class_values = []
        for i, cnt in enumerate(self._contours_dict_list):
            contour = np.array(cnt['aligned_contour']).flatten().reshape(-1, 2)
            coeffs = calculate_efd(contour, self.harmonic, self.kwargs.get('_a_n', 1),
                                   self.kwargs.get('_b_n', 1),
                                   self.kwargs.get('_c_n', 1),
                                   self.kwargs.get('_d_n', 1),
                                   True,#self.kwargs.get('_symmetric_components', False),
                                   True#self.kwargs.get('_asymmetric_components', False),
                                   )
            if self.kwargs.get('normalize', False) and self.kwargs.get('size_invariant', False):
                rotation_invariant_coeffs, dgree = normalize_efd(coeffs, size_invariant=True)
            elif self.kwargs.get('normalize', False):
                rotation_invariant_coeffs, dgree = normalize_efd(coeffs, size_invariant=False)
            elif self.kwargs.get('size_invariant', False):
                scale_invariant_coeffs = self.normalize_size_only_efd(coeffs)

            if self.kwargs.get('_asymmetric_components',False) == False :
                for pnt in coeffs:
                    pnt[0] = pnt[3] = 0
            if self.kwargs.get('_symmetric_components',False) == False :
                for pnt in coeffs:
                    pnt[1] = pnt[2] = 0


            coeffs = coeffs.flatten()
            if self.prepared_data.any():
                self.prepared_data = np.c_[self.prepared_data, coeffs]
            else:
                self.prepared_data = coeffs

            try:
                self.class_values.append(cnt['class_value'])
            except Exception as e:
                print("class_values:", e)
                tb = traceback.format_exc()
                print(tb)
                continue



    def normalize_phase(self, coeffs):
        a_1, b_1, c_1, d_1 = coeffs[0, :]

        theta_1 = 0.5 * np.arctan(2 * (a_1 * b_1 + c_1 * d_1) /
                                  np.sqrt(pow(a_1, 2) + pow(b_1, 2) + pow(c_1, 2) + pow(d_1, 2)))
        # print("theta_1=",theta_1)
        # print("coeffs.shape[0]", coeffs.shape[0])
        coeffs_star = np.zeros(coeffs.shape)

        for i in range(coeffs.shape[0]):
            # a, b, c, d = coeffs[i, :]
            i_theta = i * theta_1
            k = np.array([[np.cos(i_theta), -np.sin(i_theta)], [np.sin(i_theta), np.cos(i_theta)]])
            m = coeffs[i, :].reshape(2, 2)

            coeffs_star[i, :] = np.dot(m, k).flatten()
            # a_star_1, _, c_star_1, _ = coeffs_star[0, :]

        return coeffs_star

    def normalize_rotation(self, coeffs_star):
        a_star_1, _, c_star_1, _ = coeffs_star[0, :]
        phi_1 = np.arctan(c_star_1 / a_star_1)

        k = [[np.cos(phi_1), np.sin(phi_1)], [-np.sin(phi_1), np.cos(phi_1)]]

        coeffs_rot_inva = np.zeros(coeffs_star.shape)
        for i in range(coeffs_star.shape[0]):
            # a_2star = a_star * np.cos(phi_1) + c_star * np.sin(phi_1)
            # b_2star = b_star * np.cos(phi_1) + d_star * np.sin(phi_1)
            # c_2star = -1 * a_star * np.sin(phi_1) + c_star * np.cos(phi_1)
            # d_2star = -1 * b_star * np.sin(phi_1) + d_star * np.cos(phi_1)
            # coeffs_rot_inva[i, :] = a_2star, b_2star, c_2star, d_2star

            m = coeffs_star[i, :].reshape(2, 2)

            coeffs_rot_inva[i, :] = np.dot(k, m).flatten()
#            print("b_2star, c_2star=", coeffs_rot_inva[i, :])
        return coeffs_rot_inva

    def get_reconstructed_efd(self, point):

        eigenvector = self.pca.components_[self.components, :]

        coeffs = np.dot(point,
                        eigenvector)

        coeffs += self.mu


        coeffs = coeffs.squeeze().reshape(-1, 4)


        xt, yt = inverse_transform(coeffs, harmonic=self.harmonic)

        result = np.c_[xt, yt]

        return result

    def get_total_variance(self):
        return np.sum(self.pca.explained_variance_)

    # ============================== LDA ==========================
    def prepare_vector_lda(self):
        for cnt in self._contours_dict_list:
            c = np.array(cnt['aligned_contour']).flatten()

            if self.prepared_data.any():
                self.prepared_data = np.c_[self.prepared_data, c]
            else:
                self.prepared_data = c

            self.class_values.append(cnt['class_value'])

    def compute_lda(self):
        self.prepared_classes = np.array(self.class_values)

        x = self.prepared_data.T
        y = self.prepared_classes

        self.mu = np.mean(x, axis=0)
        solver = self.kwargs.get('lda_solver')
        shrinkage = self.kwargs.get('lda_shrinkage')
        if solver == 'eigen':
            if shrinkage:
                self.lda_clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
            else:
                self.lda_clf = LinearDiscriminantAnalysis(solver='eigen')
        else:
            self.lda_clf = LinearDiscriminantAnalysis(solver='svd')

        try:
            self.lda_clf.fit(x, y)
        except Exception as e:
            import re
            order = int(re.findall('\d+', str(e))[0])

            self.compute_pca(_n_components=order - 2)
            x = self.X_transformed

            self.lda_clf.fit(x, y)
            self._perform_pca_before_lda = True

        self.X_transformed = self.lda_clf.transform(x)


    def get_total_number_of_lda(self):
        return self.X_transformed.shape[1]

    def get_total_variance_lda(self):
        return None

    def get_predict_proba_lda(self, x):
        return self.lda_clf.predict(x)

    def get_reconstructed_lda(self, point):

        scaling = self.lda_clf.scalings_
        solver = self.kwargs.get('lda_solver')

        if solver == 'eigen':
            mt = np.linalg.pinv(scaling)

            mean_of_means = np.mean(self.lda_clf.means_, axis=0).reshape(1, -1)
            mean_transformed = self.lda_clf.transform(mean_of_means)[:, :2]

            x_hat = np.dot(np.array(point) - mean_transformed, mt[[0, 1], :]) + mean_of_means

            if self._perform_pca_before_lda:
                x_hat = self.pca.inverse_transform(x_hat)
        else:
            # for SVD Solver
            mt = np.linalg.pinv(scaling)[[0, 1], :]
            x_hat = np.dot(np.array(point), mt) + self.lda_clf.xbar_

        # print(self.lda_clf.coef_.shape)
        # # print(x_hat.shape)
        # # print("prob = ",self.lda_clf.predict_proba(x_hat.reshape(1, scaling.shape[0])))
        # pred = self.lda_clf.predict(x_hat)[0]
        # print(pred, self.lda_clf.classes_)
        # cls = list(self.lda_clf.classes_).index(pred)
        # print("cls=",cls)
        # x_hat = np.multiply(x_hat, self.lda_clf.coef_[cls, :])
        # print("hat = ",x_hat.shape)
        # #x_hat= np.dot(x_hat, coeffs[1, :])
        # #print(self.mu, self.lda_clf.xbar_)
        # #print(self.lda_clf.xbar_.shape)
        # print(x_hat.shape, np.sum(x_hat, axis=0))
        # x_hat += self.mu
        # print(np.sum(x_hat, axis=0))
        #
        # x_hat = np.dot(point,
        #                np.linalg.pinv(self.W))
        # # print(x_hat.shape, self.m)
        # x_hat += self.m
        # print("x_hat shape=", x_hat.shape)
        return x_hat

    def get_reconstructed_lda_efd(self, point):

        coeffs = self.get_reconstructed_lda(point)

        coeffs = coeffs.squeeze().reshape(-1, 4)

        xt, yt = inverse_transform(coeffs, harmonic=self.harmonic)

        result = np.c_[xt, yt]

        return result

    def comp_mean_vectors(self, X, y):
        class_labels = np.unique(y)
        n_classes = class_labels.shape[0]
        self.mean_vectors = []
        for cl in class_labels:
            self.mean_vectors.append(np.mean(X[y == cl], axis=0))
        return self.mean_vectors

    def scatter_within(self, X, y):
        class_labels = np.unique(y)
        n_classes = class_labels.shape[0]
        n_features = X.shape[1]
        mean_vectors = self.comp_mean_vectors(X, y)
        S_W = np.zeros((n_features, n_features))
        for cl, mv in zip(class_labels, mean_vectors):
            class_sc_mat = np.zeros((n_features, n_features))
            for row in X[y == cl]:
                row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
                class_sc_mat += (row - mv).dot((row - mv).T)
            S_W += class_sc_mat
        return S_W

    def scatter_between(self, X, y):
        overall_mean = np.mean(X, axis=0)
        self.m = np.mean(X, axis=0)
        n_features = X.shape[1]
        mean_vectors = self.comp_mean_vectors(X, y)
        S_B = np.zeros((n_features, n_features))
        classes = np.unique(y)
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == classes[i], :].shape[0]
            mean_vec = mean_vec.reshape(n_features, 1)
            overall_mean = overall_mean.reshape(n_features, 1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        return S_B

    def get_components2(self, eig_vals, eig_vecs, X, n_comp=2):
        n_features = X.shape[1]
        eig_pairs = [(np.abs(eig_vals[i]).real, eig_vecs[:, i].real) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        W = np.hstack([eig_pairs[i][1].reshape(n_features, 1) for i in range(0, n_comp)])
        # print("w=",W)
        return W

    def matlab_LDA(self, X, y):
        self.mean = np.mean(X, 0)
        # print("mean shape = ", self.mean.shape)
        X -= self.mean
        class_labels = np.unique(y)
        n_classes = class_labels.shape[0]

        # Initialize Sw
        n_features = X.shape[1]
        Sw = np.zeros((n_features, n_features))

        # Compute total covariance matrix
        St = np.cov(X)

        # sum over classes
        for i in range(n_classes):
            # Get all instances with class i
            current_X = X[y == class_labels[i], :]
            # print(current_X.shape)
            # update within-class scatter
            C = np.cov(current_X)
            p = current_X.shape[0] / (len(class_labels) - 1)
            # print(p, C.shape, Sw.shape)
            Sw = Sw + (p * C)

        # Compute between class scatter
        Sb = St - Sw
        Sb[np.isnan(Sb)] = 0
        Sw[np.isnan(Sw)] = 0
        Sb[np.isinf(Sb)] = 0
        Sw[np.isinf(Sw)] = 0

        # Perform eigen decomposition of inv(Sw) * Sb
        self.eig_vals, self.eig_vecs = np.linalg.eig(Sb.dot(Sw))

        self.eig_vecs[np.isnan(self.eig_vecs)] = 0
        eig_pairs = [(np.abs(self.eig_vals[i]).real, self.eig_vals[:, i].real) for i in range(len(self.eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        self.M = np.hstack([eig_pairs[i][1].reshape(n_features, 1) for i in range(0, 2)])

        self.X_transformed = X.dot(self.M)

    def export_shape_space_transformed_data_to_csv(self, dst_directory, filename):
        """
        This function export the transformed data. For instance, let X be the matrix of
        the given data, the transformed data matrix is calculated by:

        pca = PCA()
        pca.fit(X)
        X_transformed = pca.transform(X)

        :param components: numpy array
        :return:
        """
        # remove extension
        filename = filename.split('.')[0]
        data_filename = Helper.build_path(dst_directory, filename + '_transformed_data.csv')
        values_to_export = self.X_transformed
        compo_header = ''
        for i in range(1, values_to_export.shape[1] + 1):
            if i == values_to_export.shape[1]:
                compo_header += 'PCA_{}'.format(i)
                continue
            compo_header += 'PCA_{},'.format(i)
        compo_header += ', class_category'

        reflected_points_dict = self.separate_points_by_class()

        class_values = []
        for i, cnt in enumerate(self._contours_dict_list):
            c = np.array(cnt['aligned_contour']).flatten()
            if np.array_equal(self.prepared_data[:, i], c):
                try:
                    class_values.append(str(cnt['class_value']))
                except Exception:
                    tb = traceback.format_exc()
                    print(tb)
                    continue
            else:
                class_values.append('None')

        values_to_export = values_to_export.astype(str)
        if self.class_values:
            values_to_export = np.c_[values_to_export, self.class_values]
        else:
            values_to_export = np.c_[values_to_export, class_values]

        for class_number, points in reflected_points_dict.items():
            for point in points:
                for i in range(values_to_export.shape[0]):
                    if point.tolist() == values_to_export[i, self.components].tolist():
                        values_to_export[i, - 1] = str(class_number)
                        break

        np.savetxt(data_filename, values_to_export, delimiter=',', header=compo_header, comments="", fmt="%s")

    def export_shape_space_components_to_csv(self, dst_directory, filename):
        """
        This function export the components of the model. For instance, let X be the matrix of
        the given data, the components are calculated by:

        pca = PCA()
        pca.fit(X)
        compo = pca.components_ : is the orthogonal basis of the space your projecting the data into.

        :param dst_directory:
        :param filename:
        :return:
        """
        # remove extension
        filename = filename.split('.')[0]
        component_filename = Helper.build_path(dst_directory, filename + '_orthogonal_component.csv')
        mean_filename = Helper.build_path(dst_directory, filename + '_mean.csv')
        components_values_to_export = self.pca.components_
        compo_header = ''
        for i in range(1, components_values_to_export.shape[1] + 1):
            if i == components_values_to_export.shape[1]:
                compo_header += 'PCA_{}'.format(i)
                continue
            compo_header += 'PCA_{},'.format(i)

        np.savetxt(component_filename, self.pca.components_, delimiter=',', header=compo_header, comments="")

        # save mean of the data
        np.savetxt(mean_filename, self.mu, delimiter=',', header='mean', comments="")

    def export_pca_prepared_data(self, dst_directory, filename):
        """
        This function export the prepared data (aligned data + class values)

        :param dst_directory:
        :param filename:
        :param class_values:
        :return: -
        """
        # remove extension
        filename = filename.split('.')[0]
        data_filename = Helper.build_path(dst_directory, filename + '_prepared_data.csv')
        class_values = []
        for i, cnt in enumerate(self._contours_dict_list):
            c = np.array(cnt['aligned_contour']).flatten()
            if np.array_equal(self.prepared_data[:, i], c):
                try:
                    class_values.append(str(cnt['class_value']))
                except Exception:
                    tb = traceback.format_exc()
                    print(tb)
                    continue
            else:
                class_values.append('None')

        header = ''
        # fmt = ''
        for i in range(1, self.prepared_data.shape[0]):
            header += 'X_{}, '.format(i)
        header += 'X_{}, class'.format(self.prepared_data.shape[0] + 1)
        fmt = '%s'

        values_to_export = self.prepared_data.T.astype(str)

        if self.class_values:
            values_to_export = np.c_[values_to_export, self.class_values]
        else:
            values_to_export = np.c_[values_to_export, class_values]

        np.savetxt(data_filename, values_to_export, delimiter=',', header=header, comments="", fmt=fmt)

    def reconstruct_shape_on_specific_pca(self):
        """
        This function reconstruct the shapes on a specific principal component
        :return:
        """
        eigenvector = self.pca.components_
        pca_min = min(eigenvector[1, :])
        pca_max = max(eigenvector[1, :])
        portion = (pca_max - pca_min) / 20
        list_of_points = []
        for i in range(pca_min, pca_max, portion):
            list_of_points.append(np.asarray([0, i]))

        result = np.array([])
        for point in list_of_points:
            x_hat = np.dot(point, eigenvector[self.components, :])
            x_hat += self.mu

            result = np.c_[result, x_hat]

            # DetectContour.draw_reconstructed_data(
            #     x_hat, "{}/{}_pca2_{}.png".format(self.temp_directory, counter,int(i)))

        return result

    def get_shape_space_mean_shape(self):
        """
        This function create mean shape of the shapes in shape space
        :return:
        """
        eigenvector = self.pca.components_[[0, 1], :]
        x_hat = np.dot([0, 0], eigenvector)
        x_hat += self.mu

        return [x_hat.reshape(-1, 1, 2)]

    def create_shapes_based_on_std(self):
        point = np.array([0, 0])
        coordinate_values = self.get_shape_space_coordinate_values()
        std_points = np.std(coordinate_values, axis=0, ddof=1)

        # mean_shape = self.get_shape_space_mean_shape()
        # mean_shape = np.vstack(mean_shape).squeeze().reshape(-1, 2)
        # mean_shape += std_points
        # mean_shape = mean_shape.reshape(-1, 4)

        # eigenvector = self.pca.components_

        point = np.array([0, 0]) + 2 * std_points
        rec_array = self.get_reconstruct_point(point)

        point = np.array([0, 0]) - 2 * std_points
        rec_array = self.get_reconstruct_point(point)
