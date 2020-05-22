import ast
import csv
import inspect
import math
import os
import pickle
import random
import re
import shutil
import tarfile
import traceback
from glob import glob

import OpenGL.GLU as GLU
import cv2
import imutils
import pandas as pd
import pyrr
from PyQt5 import QtWidgets
from scipy.spatial import ConvexHull, KDTree

from .decorators import *


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def exit_handler(path):
        shutil.rmtree(path)
        print("Cleaned!")

    @staticmethod
    def get_image_height_width(image_path):
        """

        :param image_path:
        :param add_border_value:
        :return: height, width
        """
        try:
            # Load Image
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
        except:
            tb = traceback.format_exc()
            print(tb)
            return None, None

        return height, width

    @staticmethod
    def euclidean_distance(point1, point2):
        try:
            x1, y1, z1 = point1
        except ValueError:
            x1, y1 = point1
        try:
            x2, y2, z2 = point2
        except ValueError:
            x2, y2 = point2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        a = np.array([x1, y1])
        b = np.array([x2, y2])
        dist = np.linalg.norm(a - b)

        return dist

    @staticmethod
    # @to_contour_opencvformat
    def nearest_point(contours, point):
        """

        :param contours:
        :param x:
        :param y:
        :return:
        """
        selected_point = None
        min_dist = 1000

        try:
            cnt = contours[0]
        except IndexError:
            # self.logger.error('No contour founded', exc_info=True)
            return selected_point
        for data in cnt:
            dist = Helper.euclidean_distance(point, (data[0][0], data[0][
                1]))
            if dist < min_dist:
                selected_point = data
                min_dist = dist

        return selected_point




    @staticmethod
    def find_two_nearest_points(contours, point):
        """
        find the two nearest point to the one that we add in order to place
        the point in the correct position between points
        :param point: (x,y) coordinate
        :param contours: contours of the image
        :return:
        """
        first_point = None
        second_point = None
        first_min_dist = 1000
        second_min_dist = 1000
        # if len(contours) < 2:
        #     print("List of contours must have more than 1 element!")
        #     return 1
        # else:

        pos1 = 0
        i = 0
        for contour in contours:
            for cnt_point in contour:
                dist = Helper.euclidean_distance(point, cnt_point[0])
                i += 1
                if dist < first_min_dist:
                    first_min_dist = dist
                    first_point = cnt_point
                    pos1 = i

        pos2 = 0
        i = 0
        for contour in contours:
            for cnt_point in contour:
                dist = Helper.euclidean_distance(point, cnt_point[0])
                i += 1
                if dist < second_min_dist and not Helper.compare_equality(
                        first_point, cnt_point):
                    second_min_dist = dist
                    second_point = cnt_point
                    pos2 = i

        if pos2 < pos1:
            temp = first_point
            first_point = second_point
            second_point = temp

        return first_point, second_point

    @staticmethod
    def find_closest_line_segment(contours, point):
        """
         find closest line segment to the given point. it returns tuple of
         start and end point of the line segment.
        :param contours:
        :param point:
        :return: tuple
            (ndarray, ndarray)
        """
        first_point = None
        second_point = None
        min_dist = 1e20

        for contour in contours:
            for i in range(len(contour)):
                j = i + 1
                if j == len(contour):
                    j = 0

                p1 = np.array(contour[i][0])[:2]
                p2 = np.array(contour[j][0])[:2]
                v = p2 - p1

                ap = point - p1
                ab = p2 - p1
                q = p1 + np.dot(ap, ab) / np.dot(ab, ab) * ab

                t = np.dot(v, (q - p1)) / np.dot(v, v)

                if 0 < t < 1:
                    dist = Helper.euclidean_distance(q, point)
                    if dist < min_dist:
                        first_point = p1
                        second_point = p2
                        min_dist = dist
            if first_point is not None and second_point is not None:
                first_point = np.array([[first_point[0], first_point[1], -1.0]])
                second_point = np.array([[second_point[0], second_point[1], -1.0]])

        return first_point, second_point

    @staticmethod
    @to_contour_opencvformat
    def add_point_to_contour(contours, point):
        """
        This function will simply add the given point between two nearest
        contour points.
        :param contour: contours can be results from OpenCV findContours
        function, Or list of lists for example: [[x1, y1], [x2, y2]]
        :param point: 2d point for example: [x,y] or (x,y)
        :return:
        """
        new_contour = []
        first, second = Helper.find_two_nearest_points(contours, point)
        for cnt in contours:
            for cnt_point in cnt:
                if (first[0] == cnt_point[0]).all() or (second[0] == cnt_point[
                    0]).all():
                    new_contour.append(cnt_point[0])
                    new_contour.append(point)
                else:
                    new_contour.append(cnt_point[0])

        return new_contour

    @staticmethod
    def map_qt_to_opengl_coordinates(point_3D, viewport, width, height,
                                     model_view_matrix,
                                     projection_matrix):
        """
        Map point in qt widget coordinate to OpenGL homogeneous coordinate
        :param point_3D: point in 3 dimension ex: (x, y, z).
        for 2D simply put 'z' coordinate equal to 0.
        :param viewport: can be result from glGetIntegerv(GL_VIEWPORT)
        :param height: texture or image height
        :param width: texture or image width
        :param model_view_matrix: 4x4 model view matrix
        :param projection_matrix: 4x4 projection matrix
        :return: mapped point in 3D coordinate
        """
        x, y, z = point_3D
        view_port = viewport
        if view_port[2] == 2 * width:
            view_port[3] = int(view_port[3] / 2)
            view_port[2] = int(view_port[2] / 2)
        else:
            view_port[3] = height
            view_port[2] = width

        y_real = view_port[3] - y

        x_new, y_new, z_new = GLU.gluUnProject(x, y_real, z,
                                               model_view_matrix,
                                               projection_matrix,
                                               view_port)
        mapped_point = (x_new, y_new, z_new)

        return mapped_point

    @staticmethod
    def map_from_image_to_opengl(contours, viewport, width, height):
        """
        :param contours: contours can be results from OpenCV findContours
        function, Or list of lists
        :param viewport: can be result from glGetIntegerv(GL_VIEWPORT)
        :param width: Width of the texture (image)
        :param height: Height of the texture (image)
        :return: list of numpy array e.x: [array([[[x1, y1, z1]],[[x2, y2,
        z2]]...])]
        """
        view_port = viewport

        view_port[2] = width
        view_port[3] = height

        all_cnt_coordinates = []
        try:
            cnt = contours[0]
        except IndexError:
            # self.logger.error('No contour founded', exc_info=True)
            return all_cnt_coordinates
        coordinate = []
        for point in cnt:
            x = point[0][0]
            y = point[0][1]
            z = 0
            if x is None or y is None:
                continue
            model_view_matrix = pyrr.Matrix44.identity()
            projection_matrix = pyrr.Matrix44.identity()

            y_real = view_port[3] - y
            x_new, y_new, z_new = GLU.gluUnProject(x, y_real, z,
                                                   model_view_matrix,
                                                   projection_matrix,
                                                   view_port)
            coordinate.append([x_new, y_new, z_new])
            # coordinate.append(y_new)
            # coordinate.append(z_new)

        all_cnt_coordinates.append(np.asarray(coordinate))

        return all_cnt_coordinates

    @staticmethod
    def map_opengl_to_qt_coordinates(point_3D, viewport,
                                     model_view_matrix, projection_matrix):
        """
        Map point in OpenGL homogeneous coordinate [-1,1] to qt widget
        coordinate.
        :param point_3D:
        :param viewport:
        :param model_view_matrix:
        :param projection_matrix:
        :return:
        """
        x, y, z = point_3D
        widget_x, widget_y, widget_z = GLU.gluProject(x, y, z,
                                                      model_view_matrix,
                                                      projection_matrix,
                                                      viewport)
        point_in_widget = (widget_x, widget_y, widget_z)

        return point_in_widget

    @staticmethod
    def map_from_opengl_to_image(point_3D, viewport, width, height):
        """

        :param point_3D:
        :param width: Width of the texture (image)
        :param height: Height of the texture (image)
        :param viewport:
        :return: [x_new, y_new, z_new]
        """
        x, y, z = point_3D
        view_port = viewport
        view_port[2] = width
        view_port[3] = height

        model_view_matrix = pyrr.Matrix44.identity()
        projection_matrix = pyrr.Matrix44.identity()

        x_new, y_new, z_new = GLU.gluProject(x, y * -1, z,
                                             model_view_matrix,
                                             projection_matrix,
                                             view_port)
        mapped_point = (x_new, y_new, z_new)

        return mapped_point

    @staticmethod
    def map_contour_from_opengl_to_image2(contours, width, height, viewport):
        """

        :param width: Width of the texture (image)
        :param height: Height of the texture (image)
        :param contours:
        :return: numpy array e.x: [array([[[x1, y1, z1]],[[x2, y2, z2]]...])]
        """
        view_port = viewport
        view_port[2] = width
        view_port[3] = height
        model_view_matrix = pyrr.Matrix44.identity()
        projection_matrix = pyrr.Matrix44.identity()
        # print("view_port=", view_port)
        all_cnt_coordinates = []
        try:
            cnt = contours[0]
        except IndexError:
            # self.logger.error('No contour founded', exc_info=True)
            return all_cnt_coordinates
        coordinate = []
        for point in cnt:
            x = point[0][0]
            y = point[0][1]  # * -1
            z = point[0][2]
            x_new, y_new, z_new = GLU.gluProject(x, y, z,
                                                 model_view_matrix,
                                                 projection_matrix,
                                                 view_port)
            coordinate.append(x_new)
            coordinate.append(y_new)
            # coordinate.append(z_new)

        all_cnt_coordinates.append(np.array(coordinate).reshape((-1, 1,
                                                                 2)))

        return all_cnt_coordinates

    @staticmethod
    @to_contour_opencvformat
    def compute_centroid(cnt_points, dtype=np.float32):
        cnt_squeezed = np.vstack(cnt_points[0]).squeeze()
        x = [p[0] for p in cnt_squeezed]
        y = [p[1] for p in cnt_squeezed]

        cx = sum(x) / len(cnt_squeezed)
        cy = sum(y) / len(cnt_squeezed)

        center = [cx, cy]

        return center

    @staticmethod
    def compute_centroid_np(points, dtype=np.float32):
        """
        calculate centroid of the numpy array.
        (This is the fastest way!)
        :param points: ndarray
        :param dtype: str or dtype
        :return: (1,2) ndarray
            centroid of the points
        """
        length = points.shape[0]
        sum_x = np.sum(points[:, 0])
        sum_y = np.sum(points[:, 1])

        center = [sum_x / length, sum_y / length]

        return np.array([center], dtype=dtype)

    @staticmethod
    def choose_point_in_between(point1, point2, dist):
        """
        find a point between the two given points with the specific distance
        from first point
        :param point1:
        :param point2:
        :param dist: new point distance from first point
        :return:
        """
        x1 = point1[0][0]
        y1 = point1[0][1]
        x2 = point2[0][0]
        y2 = point2[0][1]

        angle_X_Y = math.atan2((y2 - y1), (x2 - x1))
        delta_y_X_Y = dist * math.sin(angle_X_Y)
        delta_x_X_Y = dist * math.cos(angle_X_Y)
        x_new = x1 + delta_x_X_Y
        y_new = y1 + delta_y_X_Y

        return [x_new, y_new]

    @staticmethod
    def get_boundingrect(contours, dtype=np.float32):
        rect = []
        max_area = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > max_area:
                max_area = cv2.contourArea(cnt)
                rect = cv2.boundingRect(cnt)

        return rect

    @staticmethod
    def find_landmarks(contour, width=None, height=None,
                       temp_dir=None, image_path=None,
                       dtype=np.float32):
        """
        find the extreme points along the contour
        :param contours: founded contour points from OpenCV
        :param dtype: float32 or int32
        type of the returned points
        :return: numpy array
        [right, bottom, left, top]
        """
        landmarks = np.array([], dtype=dtype)
        if temp_dir and image_path:
            landmarks = Helper.get_landmarks_from_meta_data(contour, temp_dir, image_path)
        if landmarks.size == 0:
            landmarks = Helper.find_landmarks_from_contour(contour, width, height)[0]

            if temp_dir and image_path:
                metadata = Helper.read_metadata_from_csv(temp_dir, image_path)
                metadata['landmarks'] = landmarks
                Helper.save_metadata_to_csv(temp_dir, image_path, metadata)

        # landmarks = Helper.sort_landmarks(contour, landmarks)

        return np.array([landmarks], dtype=dtype)

    @staticmethod
    def find_landmarks_from_contour(contours, width=None, height=None,
                                    old_landmarks=np.array([]), dtype=np.float32):
        landmarks = []
        try:
            cnt = contours[0]
        except IndexError:
            # self.logger.error('No contour founded', exc_info=True)
            return []
        cnt = np.asarray(cnt, dtype=np.float32)
        ext_left = cnt[cnt[:, :, 0].argmin()]
        ext_right = cnt[cnt[:, :, 0].argmax()]
        ext_top = cnt[cnt[:, :, 1].argmax()]
        ext_bot = cnt[cnt[:, :, 1].argmin()]

        if np.any(old_landmarks):
            landmarks.append(np.roll(old_landmarks, 1, axis=0))


        elif width < height:
            landmarks.append([ext_bot, ext_top])
        else:
            landmarks.append([ext_left, ext_right])

        return np.asarray(landmarks, dtype=dtype)

    @staticmethod
    def get_landmarks_from_meta_data(contours, temp_dir, image_name):
        data = Helper.read_metadata_from_csv(temp_dir, image_name)
        try:
            str_landmarks = data['landmarks']
        except KeyError:
            return np.asarray([], dtype=np.float32)

        landmarks = []

        if 'dtype' in str_landmarks:
            str_landmarks = str_landmarks.split('dtype')[0]

        for s in re.findall('[-+]?\d*\.\d+|\d+', str_landmarks):
            try:
                landmarks.append(float(s))
            except ValueError:
                pass

        try:
            landmarks = np.asarray(landmarks, dtype=np.int32).reshape((-1, 1, 2))
        except ValueError as error:
            print(error, "\n",
                  "Please make sure the landmarks in the table is correct!")
#            print("landmarks found=", landmarks)
            return np.asarray([], dtype=np.float32)

        new_landmarks = []
        for point in landmarks:
            nearest = Helper.nearest_point(contours, point[0])
            new_landmarks.append(nearest)


        ordered_landmarks = new_landmarks

        return np.asarray(ordered_landmarks, dtype=np.float32)

    @staticmethod
    def get_bounding_rect_from_meta_data(temp_dir, image_name):
        try:
            data = Helper.read_metadata_from_csv(temp_dir, image_name)
            bounding_rect = data.get('bounding_rect', '')
            if bounding_rect:
                bounding_rect = ast.literal_eval(bounding_rect)
        except FileNotFoundError as e:
            print(e)
            bounding_rect = ''

        return bounding_rect

    @staticmethod
    def flip_contour_points(contours, landmarks, dtype=np.float32):
        """
            flip the input contours (array of points)
        :param contours: ndarray
        :param dtype: str or dtype
                    data type of the
        :return: ndarray
        """
        flipped_contour = []
        flipped_landmarks = []
        contours = np.asarray(contours, dtype=np.float32)
        landmarks_new = np.asarray(landmarks, dtype=np.float32)
        try:
            cnt = contours[0]
        except IndexError as e:
            print(e)
            # self.logger.error('No contour founded', exc_info=True)
            return [], []
        cnt_new = np.asarray(cnt, dtype=np.float32)

        if True:  # Helper.need_flip(cnt, based_on='points'):
            _, _, w, h = cv2.boundingRect(cnt_new)
            if w < h:
                cnt_new = Helper.flip_contour(cnt_new, axis=0)
                landmarks_new = Helper.flip_landmarks(landmarks_new, cnt_new, axis=0)
            else:
                cnt_new = Helper.flip_contour(cnt_new, axis=1)
                landmarks_new = Helper.flip_landmarks(landmarks_new, cnt_new, axis=1)

            landmarks_new = landmarks_new.reshape(-1, 1, 2)
            flipped_landmarks.append(landmarks_new)
            flipped_landmarks = np.asarray(flipped_landmarks,
                                           dtype=dtype)[:, ::-1]
            cnt_new = cnt_new.reshape(-1, 1, 2)
            flipped_contour.append(cnt_new)
            flipped_contour = np.asarray(flipped_contour,
                                         dtype=dtype)[:, ::-1]
        else:
            cnt_new = cnt_new.reshape(-1, 1, 2)
            flipped_contour.append(cnt_new)
            landmarks_new = landmarks_new.reshape(-1, 1, 2)

            flipped_landmarks.append(landmarks_new)

        return flipped_contour, flipped_landmarks

    @staticmethod
    def need_flip(contour, based_on='points'):
        """
            This function will consider the line between two landmarks
            (top and the bottom of the leaf) and compare the number of
            points on left (Bottom) and right (top) side of the line.
        :param based_on: optional, string
            'points': flip based on number of contour points on
                      each side of the line
            'area': flip based on area
        :param contour: ndarray
            contour points
        :return: Bool
                True: if left has more points
                False: if right side has more points
        """
        cnt = np.array(contour)
        ext_left = cnt[cnt[:, :, 0].argmin()]
        ext_right = cnt[cnt[:, :, 0].argmax()]
        ext_top = cnt[cnt[:, :, 1].argmax()]
        ext_bot = cnt[cnt[:, :, 1].argmin()]
        try:
            cnt = Helper.change_number_of_contour_coordinates(
                cnt, remove_axis='z')
        except IndexError:
            pass
        _, _, w, h = cv2.boundingRect(cnt)

        if w < h:
            a, b, c = Helper.solve_line_equation([ext_top[0], ext_bot[0]])
        else:
            a, b, c = Helper.solve_line_equation([ext_left[0], ext_right[0]])

        posetive_side_points = []
        negative_side_points = []
        if based_on == 'points':
            for point in cnt:
                r = a * point[0][0] + b * point[0][1] + c
                if r >= 0:
                    posetive_side_points.append(point)
                elif r < 0:
                    negative_side_points.append(point)

            if len(posetive_side_points) >= len(negative_side_points):
                return True
            else:
                return False
        elif based_on == 'area':
            points_on_the_line = []
            min_dist = []

            for point in cnt:
                r = a * point[0][0] + b * point[0][1] + c
                if r >= 0:
                    posetive_side_points.append(point)
                elif r < 0:
                    negative_side_points.append(point)

            blank_image = np.zeros((h + 1000, w + 1000), dtype=np.int32)
            blank_image[:, :] = 255

            cv2.drawContours(blank_image, posetive_side_points, -1, (0, 0, 0), -1)

            cv2.imwrite("./asd.png", blank_image)
            posetive_side_points = np.asarray(posetive_side_points, dtype=np.float32)
            posetive_area = cv2.contourArea(posetive_side_points)

            negative_side_points = np.asarray(negative_side_points, dtype=np.float32)
            negative_area = cv2.contourArea(negative_side_points)

            if posetive_area >= negative_area:
                return True
            else:
                return False

                # for i, point in enumerate(cnt):
                #     print(i)
                #     r = a * point[0][0] + b * point[0][1] + c
                #
                #     dist = Helper.dist_to_segment(ext_top[0][0],ext_top[0][1],
                #                               ext_bot[0][0],ext_bot[0][1],
                #                               point[0][0], point[0][1])
                #     print(dist)
                #     if r >= 0 : print(r , dist)

    @staticmethod
    def dist_to_segment(ax, ay, bx, by, cx, cy):
        """
        Computes the minimum distance between a point (cx, cy) and a line segment with endpoints (ax, ay) and (bx, by).
        :param ax: endpoint 1, x-coordinate
        :param ay: endpoint 1, y-coordinate
        :param bx: endpoint 2, x-coordinate
        :param by: endpoint 2, y-coordinate
        :param cx: point, x-coordinate
        :param cy: point, x-coordinate
        :return: minimum distance between point and line segment
        """
        from math import sqrt
        # avoid divide by zero error
        a = max(by - ay, 0.00001)
        b = max(ax - bx, 0.00001)
        # compute the perpendicular distance to the theoretical infinite line
        dl = abs(a * cx + b * cy - b * ay - a * ax) / sqrt(a ** 2 + b ** 2)
        # compute the intersection point
        x = ((a / b) * ax + ay + (b / a) * cx - cy) / ((b / a) + (a / b))
        y = -1 * (a / b) * (x - ax) + ay
        # decide if the intersection point falls on the line segment
        if (ax <= x <= bx or bx <= x <= ax) and (ay <= y <= by or by <= y <= ay):
            return dl
        else:
            # if it does not, then return the minimum distance to the segment endpoints
            return min(sqrt((ax - cx) ** 2 + (ay - cy) ** 2), sqrt((bx - cx) ** 2 + (by - cy) ** 2))

    @staticmethod
    def solve_line_equation(points):
        """
            calculate the slope and y-intercept of the line between
            two points. This method solves the equation
            a x + b y + c = 0

        :param points: array_like
                        point on 2D plane
        :return:
                a: float
                    x coefficient
                b: float
                     y coefficient
                c: float
                    y-intercept
        """
        try:
            x_coords, y_coords = zip(*points)
        except ValueError as e:
            raise e

        y2, y1 = y_coords
        x2, x1 = x_coords

        a = (y2 - y1)
        b = -1 * (x2 - x1)
        c = (x2 - x1) * y1 - (y2 - y1) * x1

        # print("Line Solution is {a}x + {b}y + {c} = 0".format(a=a, b=b, c=c))

        return a, b, c

    @staticmethod
    def flip_contour(contour, axis=0):
        """
        flip the contour points based on the axis
        :param contour: ndarray
        :param axis: {0, 1} int, optional

        :return: ndarray
        """
        temp = np.vstack(contour).squeeze()

        c = Helper.compute_centroid([contour])[0][0][0]

        temp[:, axis] += 2.0 * (c[axis] - temp[:, axis])

        return temp

    @staticmethod
    def flip_landmarks(landmark, contour, axis=0):
        temp = np.vstack(landmark).squeeze()

        c = Helper.compute_centroid([contour])[0][0][0]

        temp[:, axis] += 2.0 * (c[axis] - temp[:, axis])

        return temp

    @staticmethod
    def change_number_of_contour_coordinates(contour, remove_axis='z'):
        if remove_axis == 'x':
            cnt = np.vstack(contour).squeeze()
            cnt = np.delete(cnt, 0, axis=1).reshape((-1, 1, 2))
        elif remove_axis == 'y':
            cnt = np.vstack(contour).squeeze()
            cnt = np.delete(cnt, 1, axis=1).reshape((-1, 1, 2))
        elif remove_axis == 'z':
            cnt = np.vstack(contour).squeeze()
            cnt = np.delete(cnt, 2, axis=1).reshape((-1, 1, 2))
        else:
            raise RuntimeError("Please specify axis as string 'x', 'y' or 'z'")

        return np.asarray(cnt, dtype=np.float32)

    @staticmethod
    def convert_listoflists_to_contour_format(input_list, add_to_z=None):
        input_list = np.vstack(input_list).squeeze()

        if add_to_z:
            try:
                if input_list.shape[1] == 3:
                    contour = [np.asarray(input_list)]
                    return contour
            except IndexError:
                pass
            contour = np.c_[input_list, add_to_z * np.ones(np.asarray(
                input_list).shape[0])].reshape((-1, 1, 3))
        else:
            contour = np.array(input_list).reshape((-1, 1, 2))

        contour = [np.asarray(contour)]
        return contour

    @staticmethod
    def compare_equality(point1, point2):
        return np.equal(point1, point2).all()

    @staticmethod
    def contain_list_or_array(input_list, list_of_lists):
        """
        check if list1 is in list 2 or not
        :param input_list:
        :param list_of_lists:
        :return:
        """
        if type(input_list) == np.ndarray or type(list_of_lists) == np.ndarray:
            for l in list_of_lists:
                if np.equal(input_list, l).all():
                    return True
        else:
            for l in list_of_lists:
                if type(l) == np.ndarray:
                    if np.equal(input_list, l).all():
                        return True
                        # else:
                        #     return False
                else:
                    if input_list == l:
                        return True
                        # else:
                        #     return False
        return False

    # =========== Messages =======================================
    @staticmethod
    def critical_message(message, window_title="!"):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle(window_title)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)

        msg.setDefaultButton(QtWidgets.QMessageBox.Ok)

    @staticmethod
    def warning_message(message, window_title="!"):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle(window_title)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)

        msg.setDefaultButton(QtWidgets.QMessageBox.Ok)

    @staticmethod
    def question_message(message, window_title="?"):
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

    # ============= Controller helper ===============================

    @staticmethod
    def get_list_of_csv_files(folder_path):
        files = os.listdir(folder_path)
        directory_list = [file for file in files if not os.path.isdir(
            folder_path + file)]

        extensions = ".csv"  # etc
        csv_files = [file for file in directory_list if
                     file.lower().endswith(extensions)]
        return csv_files

    @staticmethod
    def get_list_of_images(folder_path):
        if not os.path.isdir(folder_path):
            print("Error: No such directory,", folder_path)
            return []
        files = os.listdir(folder_path)
        directory_list = [file for file in files if not os.path.isdir(
            folder_path + file)]

        extensions = (".jpg", ".png", ".gif", ".jpeg", ".tif")  # etc
        images_list = [image for image in directory_list if
                       image.lower().endswith(extensions) and '_result_' not in image
                       ]
        cropped_len = len([image for image in images_list if '_cropped_' in image])
        if cropped_len == len(images_list):
            images_list = sorted(images_list,key = lambda x: int((x.rsplit("_")[-1]).rsplit(".")[0]))
        else:
            images_list.sort()
        return images_list

    @staticmethod
    def get_list_of_mgx_contour_files(folder_path):
        if not os.path.isdir(folder_path):
            print("Error: No such directory,", folder_path)
            return []
        files = os.listdir(folder_path)
        directory_list = [file for file in files if not os.path.isdir(
            folder_path + file)]

        extensions = (".csv", ".text", ".txt")  # etc
        cnt_files_list = [file for file in directory_list if
                          file.lower().endswith(extensions)]
#        print("cnt_files_list=", cnt_files_list)
        return cnt_files_list

    @staticmethod
    def separate_file_name_and_extension(file, keep_extension=False):
        if file is None:
            return
        name = os.path.splitext(os.path.basename(file))[0]
        extension = os.path.splitext(os.path.basename(file))[1]
        if keep_extension:
            name += extension
            extension = None
        return name, extension

    # ========== main getters ===================================================
    @staticmethod
    def get_parent_image_name(image_name):
        img_name, _ = Helper.separate_file_name_and_extension(
            image_name)

        parent_name = img_name
        if '_result_' in parent_name:
            parent_name = parent_name.split('_result_')[1]

        if '_lateral_' in parent_name:
            parent_name = parent_name.split('_lateral_')[0]

        if '_terminal' in parent_name:
            parent_name = parent_name.split('_terminal')[0]

        if '_cropped_' in parent_name:
            parent_name = parent_name.split('_cropped_')[0]

        return parent_name

    @staticmethod
    def get_main_image_name(image_name):
        """
        return main part of the image name,
        for example for: cardamine_cropped_1_lateral_2_right -> cardamine_cropped_1
        :param image_name:
        :return:
        """
        img_name, _ = Helper.separate_file_name_and_extension(
            image_name)

        main_name = img_name
        if '_result_' in main_name:
            main_name = main_name.split('_result_')[1]

        if '_lateral_' in main_name:
            main_name = main_name.split('_lateral_')[0]

        elif '_terminal' in main_name:
            main_name = main_name.split('_terminal')[0]

        return main_name

    @staticmethod
    def get_image_path(temp_dir, image_name, temp_data_dir_name):
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)
        if ("lateral" in image_name) or ("terminal" in image_name):
            image_path = Helper.get_split_result_leaflet_path(
                temp_dir, image_name + ext)
        elif "cropped" in image_name:
            image_path = Helper.get_cropped_result_image_path(
                temp_dir, image_name + ext)
        else:
            image_path = Helper.build_path(temp_dir,
                                           temp_data_dir_name,
                                           image_name + ext)

        return image_path

    @staticmethod
    def get_result_image_directory(temp_dir, image_name):
        """
        create directory for each image

        :param temp_dir:
        :param image_name:
        :return: example: /var/folders/9z/3sspmln55n15dqt5x7s1tt8m0000gn/T/tmpr3wmtj69/Cardamine
        """

        result_image_path = Helper.get_result_image_path_from_image_name(
            temp_dir, image_name)
        result_image_dir = os.path.dirname(result_image_path)

        if not os.path.isdir(result_image_dir):
            os.makedirs(result_image_dir)

        return result_image_dir

    @staticmethod
    def get_result_image_path(temp_dir, image_name):
        """
        create path to the image file
        :param temp_dir:
        :param image_name:
        :return:
        """
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)

        # if not ext:
        #     files = os.listdir(os.path.join(temp_dir, image_name))
        #     for file in files:
        #         if (image_name in file) and ("leaflet" not in file) \
        #                 and ("terminal" not in file) and ("cropped" not in file):
        #             image_name, ext = Helper.separate_file_name_and_extension(
        #                 file)
        #             if '_result_' in image_name:
        #                 image_name = image_name.split('_result_')[-1]
        #             break
        #
        # if ("leaflet" in image_name) or ("terminal" in image_name):
        #     result_image_path = Helper.get_split_result_leaflet_path(
        #         temp_dir, image_name + ext)
        # elif "cropped" in image_name:
        #     result_image_path = Helper.get_cropped_result_image_path(
        #         temp_dir, image_name + ext)
        # else:

        parent_image_name = Helper.get_parent_image_name(image_name)

        result_image_dir = Helper.build_path(temp_dir,
                                             '_result_' + parent_image_name)

        if not os.path.isdir(result_image_dir):
            os.makedirs(result_image_dir)

        result_image_path = Helper.build_path(result_image_dir,
                                              '_result_' +
                                              parent_image_name + ext)
        return result_image_path

    @staticmethod
    def get_cropped_image_path(temp_dir, image_name):
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)

        parent_image_name = Helper.get_parent_image_name(image_name)

        image_path = Helper.build_path(temp_dir, '_result_' + parent_image_name
                                       , 'cropped', image_name + ext)
        # check if the file exist
        # if os.path.isfile(image_path):
        #     return image_path
        # else:
        #     return None
        return image_path

    @staticmethod
    def get_cropped_result_image_path(temp_dir, image_name):
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)

        parent_image_name = Helper.get_parent_image_name(image_name)
        if '_result_' not in image_name:
            result_image_name = '_result_' + image_name
        else:
            result_image_name = image_name

        image_path = Helper.build_path(temp_dir, '_result_' + parent_image_name
                                       , 'cropped', result_image_name + ext)
        # check if the file exist
        # if os.path.isfile(image_path):
        #     return image_path
        # else:
        #     return None

        return image_path

    @staticmethod
    def get_cropped_image_directory(temp_dir, image_name):

        image_path = Helper.get_cropped_image_path(temp_dir, image_name)
        dir_path = os.path.dirname(image_path)

        # img_name, _ = Helper.separate_file_name_and_extension(
        #     image_name)
        #
        # parent_image_name = Helper.get_parent_image_name(image_name)
        # dir_path = Helper.build_path(temp_dir, '_result_' + parent_image_name
        #                              , 'cropped')
        # if not os.path.isdir(dir_path):
        #     os.makedirs(dir_path)

        return dir_path

    # @staticmethod
    # def get_destination_directory(temp_dir, image_name):
    #     """
    #     get destination directory base on the image name
    #     :param temp_dir:
    #     :param image_name:
    #     :return:
    #     """
    #     img_name, _ = Helper.separate_file_name_and_extension(image_name)
    #
    #     if (("leaflet" in image_name) or ("terminal" in image_name)) and ("cropped" in image_name):
    #         destination_dir = os.path.dirname(Helper.get_cropped_split_leaflets_path(
    #             temp_dir, image_name))
    #     elif ("leaflet" in image_name) or ("terminal" in image_name):
    #         destination_dir = Helper.get_split_leaflets_directory(
    #             temp_dir, image_name)
    #     elif '_cropped_' in img_name:
    #         destination_dir = Helper.get_cropped_image_directory(temp_dir,
    #                                                              img_name)
    #     else:
    #         destination_dir = Helper.get_result_image_directory(
    #             temp_dir, image_name)
    #
    #     if not os.path.isdir(destination_dir):
    #         os.mkdir(destination_dir)
    #
    #     # result_image_path = Helper.get_result_image_path_from_image_name(temp_dir, image_name)
    #     #
    #     # destination_dir = os.path.dirname(result_image_path)
    #     # print("=======get_destination_directory=====")
    #     # print("result_image_path==",result_image_path)
    #     # # print("img_name==", img_name)
    #     # print("destination_dir=", destination_dir)
    #     return destination_dir

    @staticmethod
    def get_split_leaflets_directory(temp_dir, image_name):
        img_name, _ = Helper.separate_file_name_and_extension(
            image_name)

        parent_image_name = Helper.get_parent_image_name(image_name)
        # if '_result_' not in img_name:
        #     img_name = '_result_' + img_name
        if "cropped" in image_name:
            dir_path = Helper.build_path(temp_dir,
                                         '_result_' + parent_image_name
                                         , 'cropped', 'split',
                                         img_name)
        else:
            dir_path = Helper.build_path(temp_dir, '_result_' + parent_image_name,
                                         'split')

        return dir_path

    @staticmethod
    def get_split_leaflets_path(directory, image_name):
        parent_image_name = Helper.get_parent_image_name(image_name)

        result_image_name = image_name
        if "cropped" in result_image_name:
            image_path = Helper.build_path(directory, '_result_' + parent_image_name
                                           , 'cropped', 'split',
                                           result_image_name)
        else:
            image_path = Helper.build_path(directory, '_result_' + parent_image_name
                                           , 'split', result_image_name)
        # check if the file exist
        # if os.path.isfile(image_path):
        #     return image_path
        # else:
        #     return None

        return image_path

    @staticmethod
    def get_split_result_leaflet_path(directory, image_name):
        parent_image_name = Helper.get_parent_image_name(image_name)
        if '_result_' not in image_name:
            result_image_name = '_result_' + image_name
        else:
            result_image_name = image_name
        image_path = Helper.build_path(directory, '_result_' + parent_image_name
                                       , 'split', result_image_name)
        # check if the file exist
        if os.path.isfile(image_path):
            return image_path
        else:
            return None

    @staticmethod
    def get_result_image_cutpoints_path(directory, image_name):
        parent_image_name = Helper.get_parent_image_name(image_name)
        image_name, ext = Helper.separate_file_name_and_extension(image_name, False)
        image_name = image_name + '_cutpoints' + ext

        if '_result_' not in image_name:
            result_image_name = '_result_' + image_name
        else:
            result_image_name = image_name
        image_path = Helper.build_path(directory, parent_image_name
                                       , 'split', result_image_name)
        # check if the file exist
        # if os.path.isfile(image_path):
        #     return image_path
        # else:
        #     return None
        return image_path

    @staticmethod
    def get_cropped_split_leaflets_path(temp_dir, image_name):

        if '_result_' in image_name:
            img_folder_name = image_name.split('_result_')[-1]
            img_folder_name, _ = Helper.separate_file_name_and_extension(img_folder_name)
        else:
            img_folder_name, _ = Helper.separate_file_name_and_extension(image_name)
        if '_lateral' in img_folder_name:
            img_folder_name = img_folder_name.split('_lateral')[0]
        if '_terminal' in img_folder_name:
            img_folder_name = img_folder_name.split('_terminal')[0]

        parent_image_name = Helper.get_parent_image_name(image_name)
        dir_path = Helper.build_path(temp_dir, '_result_' + parent_image_name
                                     , 'cropped', 'split',
                                     img_folder_name, image_name)
        return dir_path

    @staticmethod
    def get_cropped_split_result_leaflets_path(temp_dir, image_name):
        image_name, ext = Helper.separate_file_name_and_extension(image_name)
        parent_image_name = Helper.get_parent_image_name(image_name)
        dir_path = Helper.build_path(temp_dir, '_result_' + parent_image_name
                                     , 'cropped', 'split',
                                     image_name)
        return dir_path

    # =================

    @staticmethod
    def get_image_path_from_image_name(temp_dir, image_name, data_dir_name=None):
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)
        if not ext:
            ext = ''

        if '_result_' in image_name:
            image_name = image_name.split('_result_')[-1]

        if (("lateral" in image_name) or ("terminal" in image_name)) and "cropped" in image_name:
            image_path = Helper.get_cropped_split_leaflets_path(temp_dir, image_name + ext)
        elif ("lateral" in image_name) or ("terminal" in image_name):
            image_path = Helper.get_split_leaflets_path(
                temp_dir, image_name + ext)
        elif "cropped" in image_name:
            image_path = Helper.get_cropped_image_path(
                temp_dir, image_name + ext)
        else:
            image_path = Helper.build_path(temp_dir, data_dir_name,
                                           image_name + ext)

        return image_path

    # @staticmethod
    # def get_desired_result_image_path(temp_dir, image_name, selected_type):
    #     """
    #
    #     :param temp_dir:
    #     :param image_name:
    #     :param selected_type:
    #     :return:
    #     """
    #     image_name, ext = Helper.separate_file_name_and_extension(
    #         image_name)
    #     if '_result_' not in image_name.lower():
    #         result_image_name = '_result_' + image_name.lower()
    #     else:
    #         result_image_name = image_name.lower()
    #
    #     if selected_type == 'leaf':
    #         if "cropped" in result_image_name:
    #             result_image_path = Helper.get_cropped_image_path(
    #                 temp_dir, result_image_name + ext)
    #         else:
    #             result_image_path = Helper.get_result_image_path(
    #                 temp_dir, image_name + ext)
    #
    #     elif selected_type == 'lateral':
    #         if ("lateral" in result_image_name) and ("cropped" in result_image_name):
    #             result_image_path = Helper.get_cropped_split_leaflets_path(temp_dir, result_image_name + ext)
    #
    #         elif "lateral" in result_image_name:
    #             result_image_path = Helper.get_split_leaflets_path(
    #                 temp_dir, result_image_name + ext)
    #         # else:
    #         #     result_image_name = Helper.create_leaflet_name(result_image_name, 'terminal', 0)
    #         #     result_image_path = Helper.get_split_leaflets_path(
    #         #         temp_dir, result_image_name + ext)
    #
    #     elif selected_type == 'terminal':
    #         position = 'main'
    #         if ("terminal" in result_image_name) and "cropped" in result_image_name:
    #             result_image_path = Helper.get_cropped_split_leaflets_path(temp_dir, result_image_name + ext)
    #
    #         elif "terminal" in result_image_name:
    #             result_image_path = Helper.get_split_leaflets_path(
    #                 temp_dir, result_image_name + ext)
    #         else:
    #             result_image_name = Helper.create_leaflet_name(result_image_name, 'terminal', position, 0)
    #             result_image_path = Helper.get_split_leaflets_path(
    #                 temp_dir, result_image_name + ext)
    #     elif selected_type == 'leaflets':
    #
    #         result_image_path = Helper.get_split_leaflets_path(
    #             temp_dir, result_image_name + ext)
    #
    #     return result_image_path

    @staticmethod
    def get_result_image_path_from_image_name(temp_dir, image_name):
        """

        :param temp_dir:
        :param image_name:

        :return: string
        """
        image_name, ext = Helper.separate_file_name_and_extension(
            image_name)
        if '_result_' not in image_name:
            result_image_name = '_result_' + image_name
        else:
            result_image_name = image_name

        if (("lateral" in result_image_name) or ("terminal" in result_image_name)) \
                and "cropped" in result_image_name:
            result_image_path = Helper.get_cropped_split_leaflets_path(temp_dir, result_image_name + ext)

        elif ("lateral" in result_image_name) or ("terminal" in result_image_name):
            result_image_path = Helper.get_split_leaflets_path(
                temp_dir, result_image_name + ext)
        elif "cropped" in result_image_name:
            result_image_path = Helper.get_cropped_image_path(
                temp_dir, result_image_name + ext)
        else:
            result_image_path = Helper.get_result_image_path(
                temp_dir, result_image_name + ext)

        return result_image_path

    @staticmethod
    def get_or_create_image_directory_from_image_name(temp_dir, image_name, flag='get'):

        img_name, _ = Helper.separate_file_name_and_extension(image_name)

        if (("lateral" in image_name.lower()) or ("terminal" in image_name.lower())) \
                and ("cropped" in image_name.lower()):
            destination_dir = os.path.dirname(Helper.get_cropped_split_leaflets_path(
                temp_dir, image_name))
        elif ("lateral" in image_name.lower()) or ("terminal" in image_name.lower()):
            destination_dir = Helper.get_split_leaflets_directory(
                temp_dir, image_name)
        elif '_cropped_' in image_name.lower():
            destination_dir = Helper.get_cropped_image_directory(temp_dir,
                                                                 image_name)
        else:
            destination_dir = Helper.get_result_image_directory(
                temp_dir, image_name)

        if not os.path.isdir(destination_dir) and flag == 'create':
            os.mkdir(destination_dir)

        return destination_dir

    @staticmethod
    def get_or_create_image_directory(temp_dir, image_name, type=None, flag='get'):
        # if ("leaflet" in image_name) or ("terminal" in image_name) or ("cropped" in image_name):
        #     result_path = Helper.get_destination_directory(temp_dir, image_name)
        #     print("========result_path", result_path)
        # else:
        if type == 'cutpoints':
            result_path = Helper.get_result_image_cutpoints_path(temp_dir,
                                                                 image_name)
        elif type == 'cropped_split':
            # result_path = Helper.get_split_leaflets_directory(temp_dir,
            #                                                   image_name)
            img_name, _ = Helper.separate_file_name_and_extension(
                image_name)

            parent_image_name = Helper.get_parent_image_name(image_name)
            result_path = Helper.build_path(temp_dir,
                                            '_result_' + parent_image_name
                                            , 'cropped', 'split',
                                            img_name)

        elif type == 'split':
            result_path = Helper.get_split_leaflets_directory(temp_dir,
                                                              image_name)
        elif type == 'cropped':
            result_path = Helper.get_cropped_image_directory(temp_dir, image_name)
        else:
            img, _ = Helper.separate_file_name_and_extension(image_name)
            result_path = Helper.build_path(temp_dir, img)

        if flag == 'create':
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

        return result_path

    @staticmethod
    def check_file_existance(path):
        # check if the file exist
        if os.path.isfile(path):
            return path
        else:
            return None

    @staticmethod
    def create_leaflet_name(image_name, type_of_leaflet, position, number):
        image_name, _ = Helper.separate_file_name_and_extension(image_name)

        return image_name + "_{}_leaflet_{}_{}.png".format(
            type_of_leaflet, position, number)

    # =========================================================================

    @staticmethod
    def build_path(*args):
        result_path = ''
        for _, path in enumerate(args):
            try:
                result_path = os.path.join(result_path, path)
            except:
                tb = traceback.format_exc()
                print(tb)
                pass
        return result_path

    @staticmethod
    def remove_from_temp_directory(directory, filename=None, flag='all'):
        """
        remove files from temporary directory
        :param directory: The path to temporary directory
        :param filename: if file name is None it will remove all files
        :param flag: it can be, 'all' to remove any file or extension of the
        desired files (e.x. 'png', 'jpg', 'bmp', ...)
        :return:
        """
        if filename and flag == 'all':
            files = glob(Helper.build_path(directory,
                                           '{}.*'.format(filename)))
            for file in files:
                os.remove(file)
            # remove csv files
            files = glob(Helper.build_path(directory,
                                           '{}_contour*.*'.format(
                                               filename)))
            for file in files:
                os.remove(file)

        elif filename and flag != 'all':
            files = glob(Helper.build_path(directory,
                                           '{}*.{}'.format(filename, flag)))
            for file in files:
                os.remove(file)
            # remove csv files
            files = glob(Helper.build_path(directory,
                                           '{}_contour*.{}'.format(
                                               filename, flag)))
            for file in files:
                os.remove(file)
        elif not filename and flag != 'all':
            files = glob(Helper.build_path(directory,
                                           '*.{}'.format(flag)))
            for file in files:
                os.remove(file)
        else:
            files = glob(Helper.build_path(directory, '*'))
            for file in files:
                os.remove(file)

    @staticmethod
    def remove_resampled_contour_file_from_temp(directory, image_name):
        image_name, _ = Helper.separate_file_name_and_extension(image_name)

        image_path = Helper.get_result_image_path_from_image_name(directory, image_name)

        dst_directory = os.path.dirname(image_path)

        files = glob(Helper.build_path(dst_directory,
                                       '{}_resampled_result*.csv'.format(
                                           image_name)))

        for file in files:
            os.remove(file)

    @staticmethod
    def rename_image_and_csv_files(src_folder_path,
                                   dst_folder_path, base_name, prefix,
                                   suffix, image_format):
        """
        rename images in the given folder and move them to the destination
        folder directory. If the source and destination folder be the same
        it will replace the files.
        :param image_format:
        :param src_folder_path: images source folder path
        :param dst_folder_path: images destination folder path
        :param base_name: base part of the images name
        :param prefix: prefix of the images name
        :param suffix: suffix of the images name
        :return:
        """
        files = os.listdir(src_folder_path)
        csv_counter = 1
        image_counter = 1
        supported_formats = ['.png', '.jpg', '.bmp', '.tiff', '.tif', '.jpeg', '.csv']
        for filename in files:
            filename_path = Helper.build_path(src_folder_path, filename)

            if not os.path.isdir(filename_path) or not os.listdir(filename_path):
                src = Helper.build_path(src_folder_path, filename)
                dst_folder = Helper.build_path(dst_folder_path)

                if not os.path.isdir(dst_folder):
                    os.mkdir(dst_folder)

                dst = Helper.build_path(dst_folder, filename)
                shutil.copy2(src, dst)
                continue
            else:
                sub_files = os.listdir(filename_path)
            for infile_name in sub_files:
                name, extension = Helper.separate_file_name_and_extension(
                    infile_name)
                if extension in supported_formats:
                    if 'csv' in extension:
                        ending = 'contour{}{}'.format(str(csv_counter),
                                                      extension)
                        csv_counter += 1
                    else:
                        if image_format == 'default':
                            ending = str(image_counter) + extension
                        else:
                            ending = str(image_counter) + '.' + image_format

                        image_counter += 1

                    # check if there is any base name
                    if base_name and prefix and suffix:
                        new_name = '{}_{}_{}_{}'.format(prefix, base_name,
                                                        suffix, ending)
                        new_folder_name = '{}_{}_{}'.format(prefix, base_name,
                                                            suffix)
                    elif base_name and prefix and not suffix:
                        new_name = '{}_{}_{}'.format(prefix, base_name, ending)
                        new_folder_name = '{}_{}'.format(prefix, base_name)
                    elif base_name and not prefix and suffix:
                        new_name = '{}_{}_{}'.format(base_name, suffix, ending)
                        new_folder_name = '{}_{}'.format(base_name, suffix)
                    elif not base_name and prefix and suffix:
                        new_name = '{}_{}_{}_{}'.format(prefix, name,
                                                        suffix, ending)
                        new_folder_name = '{}_{}_{}'.format(prefix, name,
                                                            suffix)
                    elif base_name and not prefix and not suffix:
                        new_name = '{}_{}'.format(base_name, ending)
                        new_folder_name = '{}'.format(base_name)
                    elif not base_name and not prefix and suffix:
                        new_name = '{}_{}_{}'.format(name, suffix, ending)
                        new_folder_name = '{}_{}'.format(name, suffix)
                    elif not base_name and prefix and not suffix:
                        new_name = '{}_{}_{}'.format(prefix, name, ending)
                        new_folder_name = '{}_{}'.format(prefix, name)
                    else:
                        new_name = infile_name
                        new_folder_name = filename

                    new_folder_name += str(image_counter - 1)

                    src = Helper.build_path(src_folder_path, filename, infile_name)
                    dst_folder = Helper.build_path(dst_folder_path, new_folder_name)

                    if not os.path.isdir(dst_folder):
                        os.mkdir(dst_folder)

                    dst = Helper.build_path(dst_folder, new_name)
                    shutil.copy2(src, dst)

                    # remove file from temp folder
                    # shutil.rmtree(Helper.build_path(src_folder_path, filename))

    @staticmethod
    def change_scale_indicator_bg_color(font_array, font_color=0,
                                        background_color=255):
        # font_array[np.where(font_array != 0)] = font_color
        for i, value in enumerate(font_array):
            font_array[i] = background_color - font_array[i]

        return font_array

    @staticmethod
    def get_all_cropped_images_of_an_image(folder_path, image_name):
        files = os.listdir(folder_path)
        directory_list = [file for file in files if not os.path.isdir(
            folder_path + file)]

        extensions = (".jpg", ".png", ".gif", ".jpeg", ".tif")  # etc
        images_list = [image for image in directory_list if
                       image.lower().endswith(extensions) and '{}_cropped_'.format(image_name) in image]
        images_list = sorted(images_list,key = lambda x: int((x.rsplit("_")[1]).rsplit(".")[0]))
#        images_list.sort()
        return images_list

    @staticmethod
    def save_resampled_csv(directory, contours, name):
        """
        save resampled data to a csv file in temp folder.
        :param contours:
        :param name:
        :return:
        """
        image_name, _ = Helper.separate_file_name_and_extension(name)

        dst_directory = Helper.get_or_create_image_directory_from_image_name(
            directory, image_name)

        if not os.path.exists(dst_directory):
            os.makedirs(dst_directory)

        for i in range(len(contours)):
            resampled_cnt_path = Helper.build_path(
                dst_directory, '{}_resampled_result{}.csv'.format(image_name, i + 1))

            try:
                contour_np = np.vstack(contours[i]).squeeze()

                df = pd.DataFrame(data=contour_np)
                df.to_csv(resampled_cnt_path, sep=',', header=['x', 'y'],
                          index=False)
            except ValueError as e:
                print('Can not save resampled data. Error: ', e)
                pass

    @staticmethod
    def load_resampled_from_csv(directory, image):
        image_name, _ = Helper.separate_file_name_and_extension(image)

        dst_directory = Helper.get_or_create_image_directory_from_image_name(
            directory, image_name)

        search_term = os.path.join(dst_directory,
                                   '{}_resampled_result*.csv'.format(image_name))
        csv_files = glob(search_term)
        contours = []
        for cnt in csv_files:
            contour_np = pd.read_csv(cnt)
            # HERE reshape to get contour
            contour = np.array(contour_np).reshape((-1, 1, 2)).astype(np.float32)
            contours.append(contour)

        # if len(csv_files) <= 1:
        #     self.critical_message("Can't find more than one resampled " \
        #                          "contour. Please make sure you apply the " \
        #                          "finding contour to all images!")
        return contours

    @staticmethod
    def delete_resampled_from_csv(directory, image):
        image_name, _ = Helper.separate_file_name_and_extension(image)

        dst_directory = Helper.get_or_create_image_directory_from_image_name(
            directory, image_name)

        search_term = os.path.join(dst_directory,
                                   '{}_resampled_result*.csv'.format(image_name))
        csv_files = glob(search_term)
        contours = []
        for cnt in csv_files:
            os.remove(cnt)
        # if len(csv_files) <= 1:
        #     self.critical_message("Can't find more than one resampled " \
        #                          "contour. Please make sure you apply the " \
        #                          "finding contour to all images!")

        return contours


    @staticmethod
    def save_aligned_contour_csv(directory, contours, name):
        """
        save resampled data to a csv file in temp folder.
        :param contours:
        :param name:
        :return:
        """
        image_name, _ = Helper.separate_file_name_and_extension(name)

        dst_directory = Helper.get_or_create_image_directory_from_image_name(
            directory, image_name)

        if not os.path.exists(dst_directory):
            os.makedirs(dst_directory)

        for i in range(len(contours)):
            resampled_cnt_path = Helper.build_path(
                dst_directory, '{}_aligned_result{}.csv'.format(image_name, i + 1))
#            print(contours[i])
            contour_np = np.vstack(contours[i]).squeeze()
#            print(contour_np)
            df = pd.DataFrame(data=contour_np)
            if len(df.columns) == 3:
                df = df.drop(df.columns[2],axis=1)
            df.to_csv(resampled_cnt_path, sep=',', header=['x', 'y'],
                      index=False)

    @staticmethod
    def load_aligned_contour_csv(directory, image):
        image_name, _ = Helper.separate_file_name_and_extension(image)

        dst_directory = Helper.get_or_create_image_directory_from_image_name(
            directory, image_name)

        search_term = os.path.join(dst_directory,
                                   '{}_aligned_result*.csv'.format(image_name))
        csv_files = glob(search_term)
        contours = []
        for cnt in csv_files:
            contour_np = pd.read_csv(cnt)
            # HERE reshape to get contour
            contour = np.array(contour_np).reshape((-1, 1, 2)).astype(np.float32)
            contours.append(contour)

        return contours

    @staticmethod
    def delete_aligned_contour_csv(directory, image):
        image_name, _ = Helper.separate_file_name_and_extension(image)

        dst_directory = Helper.get_or_create_image_directory_from_image_name(
            directory, image_name)

        search_term = os.path.join(dst_directory,
                                   '{}_aligned_result*.csv'.format(image_name))
        csv_files = glob(search_term)
        contours = []
        for cnt in csv_files:
            os.remove(cnt)
        # if len(csv_files) <= 1:
        #     self.critical_message("Can't find more than one resampled " \
        #                          "contour. Please make sure you apply the " \
        #                          "finding contour to all images!")

        return contours



    @staticmethod
    def save_contours_csv(temp_dir, contours, image_name):
        img_name, _ = Helper.separate_file_name_and_extension(image_name)
        destination_dir = Helper.get_or_create_image_directory_from_image_name(
            temp_dir, img_name)
        dst_file = os.path.join(destination_dir, img_name)


        # check if there is any contour
        if contours is None:
            return
        for i in range(len(contours)):
            data_file_path = Helper.build_path(
                destination_dir, '{}_contour{}.csv'.format(img_name, i + 1))

            contour_np = np.vstack(contours[i]).squeeze()
            # Helper.save_contour_to_csv(data_file_path, contour_np)
            df = pd.DataFrame(data=contour_np)
            df.to_csv(data_file_path, sep=',', header=['x', 'y'], index=False)

    @staticmethod
    def load_contours_from_csv(temp_dir, image_name):
        img_name, _ = Helper.separate_file_name_and_extension(image_name)
        destination_dir = Helper.get_or_create_image_directory_from_image_name(
            temp_dir, img_name)

        search_term = os.path.join(destination_dir,
                                   '{}_contour*.csv'.format(img_name))

        csv_files = glob(search_term)
        contours = []

        for cnt in csv_files:
            contour_np = pd.read_csv(cnt)
            # HERE reshape to get contour
            contour = np.array(contour_np).reshape((-1, 1, 2)).astype(np.float32)
            # print(contour.shape)
            contours.append(contour)

        return contours

    @staticmethod
    def save_convexhull_to_csv(temp_dir, convexhull, image_name):
        img_name, _ = Helper.separate_file_name_and_extension(image_name)
        destination_dir = Helper.get_or_create_image_directory_from_image_name(
            temp_dir, img_name)

        # check if there is any contour
        for i in range(len(convexhull)):
            data_file_path = Helper.build_path(
                destination_dir, '{}_convexhull{}.csv'.format(img_name, i + 1))

            contour_np = np.vstack(convexhull[i]).squeeze()
            # Helper.save_contour_to_csv(data_file_path, contour_np)
            df = pd.DataFrame(data=contour_np)
            df.to_csv(data_file_path, sep=',', header=['x', 'y'], index=False)

    @staticmethod
    def load_convexhull_from_csv(temp_dir, image_name):
        img_name, _ = Helper.separate_file_name_and_extension(image_name)
        destination_dir = Helper.get_or_create_image_directory_from_image_name(
            temp_dir, img_name)
        search_term = os.path.join(destination_dir,
                                   '{}_convexhull*.csv'.format(img_name))

        csv_files = glob(search_term)
        convexhull_list = []

        for cnt in csv_files:
            contour_np = pd.read_csv(cnt)
            # HERE reshape to get contour
            contour = np.array(contour_np).reshape((-1, 1, 2)).astype(np.float32)
            # print(contour.shape)
            convexhull_list.append(contour)

        return convexhull_list

    @staticmethod
    def save_process_results(src_dir, dst_dir):
        search_term = os.path.join(src_dir, '*_results.csv')
        csv_files = glob(search_term)

        for file in csv_files:
            shutil.copy2(file, dst_dir)

    @staticmethod
    def rotate_image(image_path, angle=-90):
        image = cv2.imread(image_path)
        rotated = imutils.rotate_bound(image, angle)

        cv2.imwrite(image_path, rotated)

    @staticmethod
    def rotate_contour(image_path):
        image = cv2.imread(image_path, 0)

        fimg = cv2.flip(image, 1)

        border_size = 20
        image = cv2.copyMakeBorder(
            fimg, top=border_size, bottom=border_size, left=border_size,
            right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # _, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(image, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 20000]

        # for cnt in contours:
        #     area = cv2.contourArea(cnt)

        min_areas_arg = np.argmin(np.array(areas))

        cnts = contours[min_areas_arg]

        return [cnts]

    @staticmethod
    def rotate_landmarks(contours, temp_dir, image_path):

        landmarks = Helper.get_landmarks_from_meta_data(contours, temp_dir, image_path)

    @staticmethod
    def prepare_metadata_dict(result_image_path, threshold, approx_factor, minimum_perimeter, contour_thickness,
                              convexhull_thickness, scale, leaf_number=None, landmarks=None, approximation=False,
                              num_pruning_iter=None, min_leaflet_length=None, petiol_width=None,
                              leaflet_position=None, leaflet_number=None, num_left_leaflets=None,
                              num_right_leaflets=None, num_terminal_leaflets=None, bounding_rect=''):

        metadata_dict = dict.fromkeys(['image_width',
                                       'image_height',
                                       'image_name',
                                       'threshold',
                                       'approx_factor',
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
                                       'num_right_leaflets',
                                       'num_terminal_leaflets',
                                       'bounding_rect'])

        metadata_dict['scale(pixel/cm)'] = scale
        height, width = Helper.get_image_height_width(result_image_path)
        metadata_dict['image_width'] = width
        metadata_dict['image_height'] = height

        result_img_name, _ = Helper.separate_file_name_and_extension(result_image_path,
                                                                     keep_extension=True)
        metadata_dict['image_name'] = result_img_name
        metadata_dict['threshold'] = threshold
        if approximation:
            metadata_dict['approx_factor'] = approx_factor
        metadata_dict['minimum_perimeter'] = minimum_perimeter
        metadata_dict['contour_thickness'] = contour_thickness
        metadata_dict['convexhull_thickness'] = convexhull_thickness

        metadata_dict['image_counter'] = None
        metadata_dict['leaf number'] = leaf_number
        metadata_dict['landmarks'] = landmarks
        metadata_dict['as_class_param'] = None

        metadata_dict['num_pruning_iter'] = num_pruning_iter
        metadata_dict['min_leaflet_length'] = min_leaflet_length
        metadata_dict['petiol_width'] = petiol_width

        metadata_dict['leaflet_position'] = leaflet_position
        metadata_dict['leaflet_number'] = leaflet_number

        metadata_dict['num_left_leaflets'] = num_left_leaflets
        metadata_dict['num_right_leaflets'] = num_right_leaflets
        metadata_dict['num_terminal_leaflets'] = num_terminal_leaflets
        metadata_dict['bounding_rect'] = bounding_rect

        return metadata_dict

    @staticmethod
    def save_metadata_to_csv(temp_dir, image_name, metadata_dict):
        image_name, _ = Helper.separate_file_name_and_extension(image_name)

        destination_dir = Helper.get_or_create_image_directory_from_image_name(
            temp_dir, image_name)

        dst_file = os.path.join(destination_dir, image_name)

        metadata_filename = dst_file + '_metadata_info.csv'

        with open(metadata_filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metadata_dict.keys())
            writer.writeheader()
            writer.writerow(metadata_dict)

    @staticmethod
    def read_metadata_from_csv(temp_dir, image_name):
        if image_name is None:
            return None
        image_name, _ = Helper.separate_file_name_and_extension(image_name)

        destination_dir = Helper.get_or_create_image_directory_from_image_name(
            temp_dir, image_name)

        dst_file = os.path.join(destination_dir, image_name)

        metadata_filename = dst_file + '_metadata_info.csv'

        data_list = []
        with open(metadata_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for data in reader:
                data_list.append(data)

        return data_list[0]

    @staticmethod
    def get_image_counter_from_filename(image_name):
        image_name, _ = Helper.separate_file_name_and_extension(
            image_name)
        try:
            # number = image_name.split('_Leaf_')[1]
            number = image_name.split('_')[3]
            return [int(number)]
        except:
            pass

        return [int(s) for s in re.findall(r'\d+', image_name)]

    @staticmethod
    def get_all_metadata_files(temp_dir, image_name):
        image_name, _ = Helper.separate_file_name_and_extension(image_name)

        destination_dir = Helper.get_or_create_image_directory_from_image_name(
            temp_dir, image_name)

        # dst_file = os.path.join(destination_dir, image_name)

        search_term = os.path.join(destination_dir,
                                   '*_metadata_info.csv')
        metadata_files = glob(search_term)

        return metadata_files

    @staticmethod
    def get_image_number(temp_dir, image_name):
        image_name, _ = Helper.separate_file_name_and_extension(image_name)
        try:
            data = Helper.read_metadata_from_csv(temp_dir, image_name)
        except FileNotFoundError:
            return None
        try:
            image_number = int(data['image_counter'])
            return image_number
        except ValueError:
            return None

    @staticmethod
    def create_result_csv_file(dst_directory, results, filename='results.csv'):
        metadata_filename = Helper.build_path(dst_directory, filename)

        with open(metadata_filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())

            writer.writeheader()
            writer.writerows(results)

    @staticmethod
    def get_all_methods_of_class(selected_class):
        return inspect.getmembers(selected_class,
                                  predicate=inspect.ismethod)

    @staticmethod
    def get_random_rgb_color():
        return [random.uniform(0, 1), random.uniform(0, 1),
                random.uniform(0, 1)]

    @staticmethod
    def get_random_rgb_class_color(class_values):
        """

        :param number_of_colors:
        :return: dict
        """
        color = {}
        for key in class_values:
            color[str(key)] = [random.uniform(0, 1), random.uniform(0, 1),
                               random.uniform(0, 1)]

        return color

    @staticmethod
    def get_class_from_metadata(temp_dir, image_name):
        """
        use leaf number as as class number
        :param temp_dir: (string)
        :param image_name: (string)
        :return:
            (tuple) (class_name, class_value)
        """
        data = Helper.read_metadata_from_csv(temp_dir, image_name)

        try:
            if data['as_class_param']:
                return data['as_class_param'], data[data['as_class_param']]
            else:
                return '', ''
        except ValueError:
            return '', ''

    @staticmethod
    def initialize_image_dict_from_list(image_list, dict_of_leaflets, on_leaflets=False):
        counter = 1
        image_dict = {}

        if on_leaflets:
            for image in image_list:
                leaflets = dict_of_leaflets[image]
                if leaflets:
                    for leaflet in leaflets:
                        image_dict[counter] = leaflet
                        counter += 1
                else:
                    image_dict[counter] = image
                    counter += 1
        else:
            for image in image_list:
                image_dict[counter] = image
                counter += 1

        return image_dict

    @staticmethod
    def check_dict_validation(image_dict):
        if type(image_dict) != dict:
            raise ValueError('The input is not a dictionary!')

        for counter in range(1, len(image_dict) + 1):
            try:
                _ = image_dict[counter]
            except KeyError:
                return False

        return True

    @staticmethod
    def reorder_dict_keys(image_dict):
        counter = 1
        temp_dict = {}
        for _, value in image_dict.items():
            temp_dict[counter] = value
            counter += 1

        return temp_dict

    @staticmethod
    def delete_image_and_files(directory, image_name):
        image_name, _ = Helper.separate_file_name_and_extension(
            image_name)

        if (("lateral" in image_name) or ("terminal" in image_name)) and "cropped" in image_name:
            image_path = Helper.get_image_path_from_image_name(directory, image_name)

            dst_directory = os.path.dirname(image_path)
            try:
                shutil.rmtree(dst_directory)
                os.rmdir(dst_directory)
            except FileNotFoundError:
                pass

        if 'cropped' in image_name.lower():
            parent_image_name = Helper.get_parent_image_name(
                image_name)
            dst_directory = Helper.get_cropped_image_directory(
                directory, parent_image_name)
            if '_result_' in image_name.lower():
                image_name = image_name.split('_result_')[1]
            # check if exist delete the files
            search_term = os.path.join(dst_directory,
                                       '*{}.*'.format(image_name))

            contour_files = glob(search_term)
            for file in contour_files:
                # if '_result_' in os.path.basename(file):
                name, _ = Helper.separate_file_name_and_extension(file)
                Helper.remove_from_temp_directory(dst_directory,
                                                  filename=name)
            # delete other files (if exists any)
            search_term = os.path.join(dst_directory,
                                       '*{}_*.*'.format(image_name))
            contour_files = glob(search_term)
            for file in contour_files:
                # if '_result_' in os.path.basename(file):
                name, _ = Helper.separate_file_name_and_extension(file)
                Helper.remove_from_temp_directory(dst_directory,
                                                  filename=name)

        elif ('lateral' in image_name.lower()) or ('terminal' in image_name.lower()):
            # parent_image_name = Helper.get_parent_image_name(
            #     image_name)
            # dst_directory = Helper.get_split_leaflets_directory(
            #     directory, parent_image_name)
            # if '_result_' in image_name.lower():
            #     image_name = image_name.split('_result_')[1]
            # # check if exist delete the files
            # search_term = os.path.join(dst_directory,
            #                            '*{}*.*'.format(image_name))
            #
            # contour_files = glob(search_term)
            # for file in contour_files:
            #     if '_result_' in file:
            #         name, _ = Helper.separate_file_name_and_extension(file)
            #         Helper.remove_from_temp_directory(dst_directory,
            #                                           filename=name)
            result_image_path = Helper.get_result_image_path_from_image_name(
                directory, image_name)
            result_directory = os.path.dirname(result_image_path)
#            print("removed dir=", result_directory)
            try:
                shutil.rmtree(result_directory)
            except FileNotFoundError:
                pass

        else:
            # dst_directory = os.path.join(directory, image_name)
            result_directory = os.path.dirname(Helper.get_result_image_path_from_image_name(
                directory, image_name))
#            print("result_directory=", result_directory, directory)
            if os.path.exists(result_directory):
                shutil.rmtree(result_directory)

    @staticmethod
    def delete_resized_image(directory):
        search_term = os.path.join(directory,
                                   'resized_*')
        images = glob(search_term)
        for file in images:
            os.remove(file)
            print("deleted:", file)

    @staticmethod
    def closest_point_in_matrix(matrix, point):
        """

        :param contours:
        :param x:
        :param y:
        :return:
        """
        selected_point = None
        min_dist = 100
        counter = 0
        sample_num = 0
        for m_point in matrix:
            dist = Helper.euclidean_distance(point, m_point)
            if dist < min_dist:
                selected_point = m_point
                sample_num = counter
                min_dist = dist

            counter += 1

        return selected_point, sample_num

    @staticmethod
    def make_tarfile(output_filename, source_dir, compression_type="gz"):
        mode = "w:{}".format(compression_type)
        with tarfile.open(output_filename, mode) as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

    @staticmethod
    def extract_tarfile(input_filename, extracting_folder, compression_type="gz"):
        mode = "r:{}".format(compression_type)
        try:
            with tarfile.open(input_filename, mode) as tar:
                root_folder = tar.getmembers()[0]
                for x in tar.getmembers():
                    if x != root_folder:
                        new_name = "/".join(x.name.strip("/").split('/')[1:])
                        x.name = new_name

                members_filter = [x for x in tar.getmembers() if x != root_folder]
                tar.extractall(path=extracting_folder, members=members_filter)
        except:
            Helper.critical_message("Can not open the file!")

    @staticmethod
    def find_shape_space_convex_hull(points):
        hull = ConvexHull(points)

        return hull

    @staticmethod
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    @staticmethod
    def calculate_std(x, y, ellipse_scale=1):
        cov = np.cov(x, y)
        vals, vecs = Helper.eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * ellipse_scale * np.sqrt(vals)

        return w, h, theta

    @staticmethod
    def calculate_std_err(x, y,ellipse_scale=1):
        cov = np.cov(x, y)
        vals, vecs = Helper.eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * ellipse_scale * np.sqrt(vals) / np.sqrt(len(x))

        return w, h, theta

    @staticmethod
    def remove_contours_between_two_selected(temp_directory, selected_points, image_path):
        """
        This function will remove the points between the first and the
        second selected point in counter-clockwise order.

        :param selected_points:
        :param contours:
        :return:
        """
        left_contours = []
        right_contours = []
        result_path = Helper.get_result_image_path_from_image_name(temp_directory, image_path)

        first, second = [p[:-1] for p in selected_points]
        try:
            # contours = Helper.load_resampled_from_csv(temp_dir, result_path)
            # if not contours:
            contours = Helper.load_contours_from_csv(temp_directory, result_path)
        except ValueError:
            Helper.critical_message("can not find the contour!")
            return []

        first = Helper.nearest_point(contours, first)
        second = Helper.nearest_point(contours, second)

        try:
            cnt = contours[0]
        except IndexError as e:
            print(e)
            # self.logger.error('No contour founded', exc_info=True)
            return [left_contours]
        i = 0
        done = False
        while not done:
            if i == len(cnt):
                i = 0
            if (first[0] == cnt[i]).all():
                left_contours.append(cnt[i])
                while True:
                    i += 1
                    if i == len(cnt):
                        i = 0
                    left_contours.append(cnt[i])
                    if (second[0] == cnt[i]).all():
                        break
            else:
                i += 1
                continue

            right_contours.append(cnt[i])
            while True:
                i += 1
                if i == len(cnt):
                    i = 0
                right_contours.append(cnt[i])
                if (first[0] == cnt[i]).all():
                    done = True
                    break
        # right_contours = np.array(right_contours, dtype=np.float32)
        left_contours = np.array(left_contours, dtype=np.float32)
        # dist = Helper.euclidean_distance(first[0], second[0])
        # print(dist)
        # each_side_length = int(len(right_contours) / 2)
        # print(each_side_length)
        # i = 1
        # while i <= each_side_length:
        #     print(i , 2*each_side_length -i)
        #     d = Helper.euclidean_distance(right_contours[i][0], right_contours[2*each_side_length -i][0])
        #     print(d)
        #     i+=1

        return [left_contours]

    @staticmethod
    def update_landmarks(contour, temp_dir, image_name):
        # image_name, _ = Helper.separate_file_name_and_extension(image_name)
        #
        # dst_directory = Helper.get_destination_directory(directory,
        #                                                  image_name)
        # landmarks = Helper.find_landmarks(contour)
        data = Helper.read_metadata_from_csv(temp_dir, image_name)
        # landmarks = Helper.change_number_of_contour_coordinates(landmarks)
        data['landmarks'] = None

        Helper.save_metadata_to_csv(temp_dir, image_name, data)

    # @staticmethod
    # def sort_landmarks(contour, landmarks):
    #     org_shape = landmarks.shape
    #     if org_shape[-1] != contour[0].shape[-1]:
    #         print("not the same shape!")
    #         return
    #
    #     landmarks = landmarks.reshape(-1, org_shape[2])
    #
    #     landmarks_sorted = []
    #     for cnt in contour:
    #         for point in cnt.astype(np.float32).tolist():
    #             if point[0] in landmarks.astype(np.float32).tolist():
    #                 landmarks_sorted.append(point[0])
    #
    #     landmarks_sorted = np.array(landmarks_sorted, dtype=np.float32).reshape(-1, 1, org_shape[-1])
    #
    #     return landmarks_sorted

    @staticmethod
    def get_leaflet_labels_from_name(image):
        image_to_labels_dict = {}

        main_name, _ = Helper.separate_file_name_and_extension(image, keep_extension=False)
        main_name = main_name.lower()
        if '_terminal_leaflet' in main_name:
            num = main_name.split('_')[-1]
            position = main_name.split('_')[-2]
            image_to_labels_dict = {'leaflet_position': position, 'number': num}

        elif '_lateral_leaflet' in main_name:
            num = main_name.split('_')[-1]
            position = main_name.split('_')[-2]
            image_to_labels_dict = {'leaflet_position': position, 'number': num}

        return image_to_labels_dict

    @staticmethod
    def clear_data_and_temp_directories(data_directory, temp_directory):
        search_term = os.path.join(data_directory, '*')

        contour_files = glob(search_term)
        for file in contour_files:
            Helper.remove_from_temp_directory(data_directory,
                                              filename=file)

        if os.path.exists(temp_directory):
            shutil.rmtree(temp_directory)

    @staticmethod
    def write_pickle_object(filename, object):
        with open(filename, 'wb') as file:
            pickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle_object(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        return data

    @staticmethod
    def are_terminal_cut_points(paar_points, part_of_main_rachis,
                                image_width, image_height):
        point = part_of_main_rachis[0]
        left_list = []
        right_list = []
        for pp in paar_points:
            if image_width > image_height:
                if pp[1] - point[1] == 0:
                    continue
                elif pp[1] - point[1] > 0:
                    left_list.append(pp)
                else:
                    right_list.append(pp)

            else:
                if pp[0] - point[0] == 0:
                    continue
                elif pp[0] - point[0] > 0:
                    left_list.append(pp)
                else:
                    right_list.append(pp)

        if len(left_list) == 1 and len(right_list) == 1:
            return True
        else:
            return False

    @staticmethod
    def get_closest_point(all_points, point):
        all_points = np.vstack(all_points).squeeze()

        tree = KDTree(all_points)

        res_dist, res_ind = tree.query(point)
        res_ind = np.unique(res_ind)

        closest_point = all_points[res_ind]

        return closest_point, res_ind
