import logging
import math
import os
from functools import cmp_to_key
import cv2
import numpy as np

from .helper import Helper


class DetectContour:
    logger = logging.getLogger(__name__)

    def __init__(self, image_path):
        self.ImagePath = image_path
        # resized image will get a value in prepare_image
        # function.
        self.scaled_image = None
        # width of the scaled image
        self.image_width = 2200
        # we draw the contour on result_image
        self.result_image = None
        # center of the image base on the detected contour
        self.center_X = 0
        self.center_Y = 0

    @staticmethod
    def fit_imagesize(image_path, size):
        """
        Resize the image according to the given width

        return: scaled image
        """
        # Load Image
        image = cv2.imread(image_path)
        img_width = image.shape[1]
        img_height = image.shape[0]
        #print(img_width,img_height)
        
        if img_height < img_width and img_width > size:
            # keep the ratio
            r = float(size) / img_width
            # (width, height)
            dim = (size, int(img_height * r))
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        elif img_height > img_width and img_height > size:
            # keep the ratio
            r = float(size) / img_height
            # (width, height)
            dim = (int(img_width * r), size)
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        else:
            return None

    def prepare_image(self, image_path, thresh, max_value, bounding_rect):

        # scale the image if it has less than 1600 pixel
        if self.scaled_image is None:
            self.scaled_image = cv2.imread(image_path)
            # self.scaled_image = DetectContour.fit_imagesize(
            #     image_path, self.image_width)
            if self.scaled_image is None:
                return None

        if bounding_rect:
            self.scaled_image = self.get_shape_region(bounding_rect)

        b, g, r = cv2.split(self.scaled_image)

        # sample_contour

        # Basic threshold example
        th, thresh = cv2.threshold(b, thresh, max_value,
                                   cv2.THRESH_BINARY)
        # thresh = cv2.adaptiveThreshold(imgray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                       cv2.THRESH_BINARY, 11, 2)

        # edged = cv2.Canny(thresh, 50, 200)
        # plt.imshow(edged, cmap='gray')
        # plt.title('Canny edge detection')
        # plt.show()
        return thresh

    @staticmethod
    def contour_is_square(contour):
        """
        check if the given contour is square or not
        resturn: bool (True/False)
        """
        # approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)

        return len(approx) == 4

    def find_leaf_contour(self, thresh, max_value, min_peri=500,
                          max_peri=7000, remove_squares=True, bounding_rect=None, bordersize=5):
        """
        return list of selected contours
        """
        self.logger.info('find leaf contour...')
        if os.path.isfile(self.ImagePath):
            thresh = self.prepare_image(self.ImagePath, thresh, max_value, bounding_rect)
            if thresh is None:
                return None
        else:
            print("file {} not found".format(self.ImagePath))
            return []

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)

        # copy the scaled image into result_image
        self.result_image = self.scaled_image.copy()
        # make the background result image white
        self.result_image.fill(255)

        i = 0
        # max_area = 0
        area_list = []
        peri_list = []
        for cnt in contours:
            if not self.contour_is_square(cnt) and remove_squares:
                area_list.append(cv2.contourArea(cnt, True))
                peri_list.append(cv2.arcLength(cnt, True))
            else:
                area_list.append(cv2.contourArea(cnt, True))
                peri_list.append(cv2.arcLength(cnt, True))

        founded_contours = []
        for cnt in contours:
#            i += 1
            x1,y1,w,h = cv2.boundingRect(cnt)
            x2 = x1+w
            y2 = y1+h
            x1-=bordersize
            x2+=bordersize
            y1-=bordersize
            y2+=bordersize
            row, col = self.scaled_image.shape[:2]
#Eliminate contours too close to the image border
#Seems the entire hierarchy is passed through - eliminating 
#the "parent" image seems to eliminate children images
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt, True)
            if (x1<0 or y1<0 or x2>(col-1) or y2>(row-1)) and w!=row and h!=col: 
#                continue
                perimeter = 0
                area = 0

            if remove_squares:
                if not self.contour_is_square(cnt) and \
                        (max_peri > perimeter > min_peri) \
                        and (area > 0):
                    founded_contours.append(cnt)
            else:
                if (max_peri > perimeter > min_peri) \
                        and (area > 0):
                    founded_contours.append(cnt)
        
        if len(founded_contours) == 0:
            return None
        founded_contours = sorted(founded_contours, key=cmp_to_key(self.comp_for_cnt))
        return founded_contours


#Comparison operator to sort contours based on relative position in the parent image
    def comp_for_cnt(self,item1,item2):
        x11,y11,w1,h1 = cv2.boundingRect(item1)
        x21,y21,w2,h2 = cv2.boundingRect(item2)
        x12 = x11+w1
        y12 = y11+h1
        x22 = x21+w2
        y22 = y21+h2
        if y11>y22:
            return 1
        if y21>y12:
            return -1
        if x11>x22:
            return 1
        if x21>x22:
            return -1
        M1 = cv2.moments(item1)
        M2 = cv2.moments(item2)
        cx1 = M1['m10']/M1['m00']
        cx2 = M2['m10']/M2['m00']
        if cx1>cx2:
            return 18
        if cx2>cx1:
            return -18
        return 0

    def draw_convexhull(self, contours, line_thickness):
        """
        draw the image convex hull
        """
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.polylines(self.result_image, [hull], True, (255, 0, 0),
                          line_thickness)

    def find_convex_hull(self, contours):
        try:
            cnt = contours[0]
            hull = cv2.convexHull(cnt)
            return hull
        except Exception as e:
            print(e)
            return None

    def draw_contour(self, contours, line_thickness, interior=False):
        """
        draw the contour of the image
        """
        for cnt in contours:
            if interior == False:
                cv2.drawContours(self.result_image, [cnt], 0, (0, 0, 0),
                                 line_thickness)
            else:
                cv2.drawContours(self.result_image, [cnt], 0, (0, 0, 0),
                                 -1)
                cv2.drawContours(self.result_image, [cnt], 0, (255, 255, 255),
                                 1)

    def contour_points_approx(self, contours, factor):
        """
        approximate a contour shape to with less number of vertices
        :param contours:
        :param factor:
        :return:
        """

        for cnt in contours:
            # arclength - The function computes a curve length or a
            # closed contour perimeter
            epsilon = factor / 1000 * cv2.arcLength(cnt, True)
            # epsilon – Parameter specifying the approximation accuracy.
            # This is the maximum distance between the original curve and
            # its approximation.
            approx = cv2.approxPolyDP(cnt, epsilon, True)

        return [approx]

    @staticmethod
    def sample_contour(contours, number_of_points_inbetween,
                       temp_dir=None, image_path=None, landmarks=None):
        """
        re-sample the contour to the given number of points
        :param contours: list of contours
        :param number_of_points_inbetween: number of points between every two
        landmarks.
        :param landmarks: ndarray
        :return: re-sampled contour points
        """
        # DetectContour.logger.info('Sample contour')
        all_contours = []
        if number_of_points_inbetween < 0:
            number_of_points_inbetween = 0

        try:
            cnt = contours[0]
            _, _, w, h = cv2.boundingRect(cnt)
            cnt = cnt.astype(np.float32).tolist()
            portion_distance = []
            portions_count = 0
            l_counter = 0
            i = 0

            if landmarks is None:
                landmarks = Helper.find_landmarks(contours,
                                                  width=w,
                                                  height=h,
                                                  temp_dir=temp_dir,
                                                  image_path=image_path)

                landmarks = landmarks[0].tolist()

            else:
                try:
                    landmarks = landmarks.tolist()
                except AttributeError:
                    pass

            new_contour = []
            # calculate each portions size
            while True:
                if portions_count == len(landmarks):
                    break
                if i == len(cnt):
                    i = 0
                if cnt[i] == landmarks[l_counter]:
                    landmarks_distance = 0
                    j = i + 1
                    while True:
                        if j == len(cnt):
                            j = 0
                        landmarks_distance += Helper.euclidean_distance(
                            cnt[j - 1][0],
                            cnt[j][0])

                        if l_counter + 1 == len(landmarks):
                            l_counter = -1
                        if cnt[j] == landmarks[l_counter + 1]:
                            l_counter += 1
                            portions_count += 1
                            break
                        j += 1
                    portion_distance.append(landmarks_distance / (
                        number_of_points_inbetween + 1))

                    i = j
                else:
                    i += 1

            # find points between each pair of landmarks
            portion_counter = 0
            l_counter = 0
            i = 0
            while True:
                total_length = 0
                if portion_counter == len(landmarks): #For Open contours change to len(landmarks)-1
                    break
                if i == len(cnt):
                    i = 0
                if cnt[i] == landmarks[l_counter]:
                    new_contour.append(cnt[i])
                    points_counter = 0
                    j = i + 1
                    prev_point = cnt[j - 1]

                    while True:
                        if j == len(cnt):
                            j = 0
                        if points_counter == number_of_points_inbetween:
                            l_counter += 1
                            if l_counter == len(landmarks):
                                l_counter = 0
                            portion_counter += 1
                            break

                        length = Helper.euclidean_distance(prev_point[0],
                                                           cnt[j][0])

                        total_length += length
                        if total_length >= portion_distance[portion_counter]:
                            total_length -= length
                            # r is the distance between current point and the
                            # new point which we want to add
                            r = abs(portion_distance[portion_counter] - total_length)

                            new_contour_point = Helper.choose_point_in_between(
                                prev_point, cnt[j], r)
                            new_contour_point = np.asarray(
                                [new_contour_point], dtype=np.float32)

                            prev_point = new_contour_point
                            if np.array(prev_point).tolist() not in \
                                    np.array(landmarks).tolist():
                                new_contour.append(prev_point)

                            total_length = 0
                            points_counter += 1

                        else:
                            j += 1
                            prev_point = cnt[j - 1]

                else:
                    i += 1

            all_contours.append(new_contour)
        except IndexError:
            DetectContour.logger.error('No contour founded', exc_info=True)

        new_points = np.asarray(all_contours, dtype=np.float32)

        return new_points

    def sort_contours(self, cnt):
        return (math.atan2(cnt['x'] - self.center_X, cnt['y'] -
                           self.center_Y) + 2 * math.pi) % (2 * math.pi)

    def convert_contour_to_dict(self, contour):
        cnt_list = []
        for data in contour:
            cnt_list.append({'x': int(data[0][0]), 'y': int(data[0][1])})
        return cnt_list

    def convert_contour_from_dict(self, cnt_dict):
        cnt_list = []
        for data in cnt_dict:
            cnt_list.append([[data['x'], data['y']]])
        cnt_list = [cnt_list]
        cnt_np = np.asarray(cnt_list, dtype='int32')
        return cnt_np

    def sort_contour_points(self, contours):
        """
        sort the given contour points clockwise
        :param contours: given contours
        :return: sorted contours
        """
        cnt = contours[0]
        M = cv2.moments(cnt)
        self.center_X = int(M["m10"] / M["m00"])
        self.center_Y = int(M["m01"] / M["m00"])
        cnt_list = self.convert_contour_to_dict(cnt)

        converted_to_contour_format = self.convert_contour_from_dict(cnt_list)

        return converted_to_contour_format

    @staticmethod
    def find_image_bounding_rect(contours):
        """
        find bounding rectangles in images
        :param contours: list of founded contour of the image
        :return: list of rectangles
        """
        bounding_rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_rectangles.append((x, y, w, h))

        return bounding_rectangles

    def draw_bounding_rect(self, bounding_rectangles):
        """
        draw the founded rectangles on the image
        :param bounding_rectangles: list of bounding rectangles
        :return:
        """
        for box in bounding_rectangles:
            x, y, w, h = box
            cv2.rectangle(self.result_image, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

    def draw_min_area_rect(self, contours):
        """
        draw minimum bounding rectangle of the contours on image
        :param contours: founded contour points
        :return:
        """
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.result_image, [box], 0, (0, 0, 255), 2)

    def get_shape_region(self, bounding_rectangles, bordersize=5):
        region = None
        for i in range(len(bounding_rectangles)):
            x, y, w, h = bounding_rectangles[i]
            region = self.scaled_image[y - bordersize:y + h + bordersize,
                     x - bordersize:x + w + bordersize]
        return region

    def crop_image(self, bounding_rectangles, save_path, bordersize=5):
        """
        crop the image according to the founded rectangles
        :param bounding_rectangles: tuples of the form (x, y, width, height)
        :param save_path: string
        x and y are top-left coordinate
        :return:
        """
        image_name, ext = self.ImagePath.split('/')[-1].split('.')

        for i in range(len(bounding_rectangles)):
            x, y, w, h = bounding_rectangles[i]
            region = self.scaled_image[y - bordersize:y + h + bordersize,
                     x - bordersize:x + w + bordersize]

            cv2.imwrite(save_path + '/{}_cropped_{}.{}'.format(image_name, i + 1, ext), region)

    def crop_result_image(self, bounding_rectangles, save_path, bordersize=5):
        """
        crop the image according to the founded rectangles
        :param bounding_rectangles: tuples of the form (x, y, width, height)
        x and y are top-left coordinate
        :return:
        """
        image_name, ext = self.ImagePath.split('/')[-1].rsplit('.',1)

        for i in range(len(bounding_rectangles)):
            x, y, w, h = bounding_rectangles[i]
            region = self.result_image[y - bordersize:y + h + bordersize,
                     x - bordersize:x + w + bordersize]
            cv2.imwrite(save_path + '/_result_{}_cropped_{}.{}'.format(image_name, i + 1, ext), region)

    def save_image(self, image_path):
        """
        write the image to disk
        :param image_path: destination path of the image
        :return:
        """
        cv2.imwrite(image_path, self.result_image)

    @staticmethod
    def save_resized_image(image_path, image):
        cv2.imwrite(image_path, image)

    @staticmethod
    def find_landmarks(contours, dtype=np.float32):
        """
        find the extreme points along the contour
        :param contours: founded contour points from OpenCV
        :param dtype: float32 or int32
        type of the returned points
        :return: numpy array
        [right, bottom, left, top]
        """
        landmarks = []
        try:
            cnt = contours[0]
            cnt = np.array(cnt)
            ext_left = cnt[cnt[:, :, 0].argmin()]
            ext_right = cnt[cnt[:, :, 0].argmax()]
            ext_top = cnt[cnt[:, :, 1].argmin()]
            ext_bot = cnt[cnt[:, :, 1].argmax()]
            # landmarks.append([ext_right, ext_top, ext_left, ext_bot])
            _, _, w, h = cv2.boundingRect(cnt)
            if w < h:
                landmarks.append([ext_top, ext_bot])
            else:
                landmarks.append([ext_left, ext_right])
        except IndexError:
            DetectContour.logger.error('No contour founded', exc_info=True)

        return np.array(landmarks, dtype=dtype)

    def add_border_to_scaled_image(self, border_size=10):
        """
        adding border to scaled image in order to detect the contours that
        touches the image border
        :return:
        """
        row, col = self.scaled_image.shape[:2]
        bottom = self.scaled_image[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        self.scaled_image = cv2.copyMakeBorder(
            self.scaled_image, top=border_size, bottom=border_size, left=border_size,
            right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def remove_border_from_scaled_image(self, border_size=10):
        """
        remove the border that added before to the scaled image
        :return:
        """

        row, col = self.scaled_image.shape[:2]

        self.scaled_image = self.scaled_image[border_size:row - border_size,
                            border_size:col - border_size]

    @staticmethod
    def draw_and_save_contour(image, contours, save_path):

        cv2.drawContours(image, contours, -1, (0, 0, 0), -1)

        cv2.imwrite(save_path, image)

    @staticmethod
    def draw_reconstructed_data(reconstructed_data, image_path):
        cnt = np.array(reconstructed_data, dtype=np.int32).reshape((-1, 1, 2))

        bounding_rectangles = DetectContour.find_image_bounding_rect([cnt])
        x, y, w, h = bounding_rectangles[0]

        ext_left_x, ext_left_y = cnt[cnt[:, :, 0].argmin()][0]
        ext_top_x, ext_top_y = cnt[cnt[:, :, 1].argmin()][0]

        cnt_shifted = np.hstack([cnt[:, :, 0] - int(ext_left_x), cnt[:, :, 1] - int(ext_top_y)])
        cnt = cnt_shifted.reshape((-1, 1, 2))

        blank_image = np.zeros((h + 10, w + 10), dtype=np.int32)
        blank_image[:, :] = 255
        # DetectContour.sample_contour()
        image = cv2.drawContours(blank_image, [cnt], -1, (0, 0, 0), -1)

        # DetectContour.draw_and_save_contour(blank_image, [cnt], image_path)

        return image

    def detect_barcode(self, image_path):
        # load the image and convert it to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # blur and threshold the image
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        im2, cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        # draw a bounding box arounded the detected barcode and display the
        # image
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        cv2.imwrite("barcode.png", image)

    def change_color_of_leaf_main_image(self, image_path, contour, save_path):
        image_name, ext = self.ImagePath.split('/')[-1].rsplit('.',1)
        org_image = cv2.imread(image_path)
        for i in range(len(contour)):
            image = org_image.copy()
            cnt = contour[i]
            cv2.drawContours(image, [cnt], 0, (0, 0, 255), -1)
            cv2.imwrite(save_path + '/{}_cropped_{}.{}'.format(image_name, i + 1, ext), image)
