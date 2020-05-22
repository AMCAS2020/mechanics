import random
import traceback

import OpenGL.GL.shaders
import OpenGL.GLU as GLU
import pyrr
from OpenGL.GL import *
from PIL import Image
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QGuiApplication
from PyQt5.QtWidgets import QOpenGLWidget

from .decorators import *
from .helper import Helper
from .procrustes_analysis_worker import ProcrustesAnalysisWorkerThread


class EditContourOpenGLWidget(QOpenGLWidget):
    VERTEX_SHADER = """
        #version 330

         layout(location = 0) in vec3 position;
         layout(location = 1) in vec3 color;
         layout(location = 2) in vec2 inTexCoords;
         uniform mat4 mvp_matrix;
         out vec3 newColor;
         out vec2 outTexCoords;
        void main()
        {

            gl_Position = mvp_matrix * vec4(position, 1.0f);
            newColor = color;
            outTexCoords = inTexCoords;
        }
        """
    FRAGMENT_SHADER_TEX = """
        #version 330
        in vec2 outTexCoords;

        out vec4 tex;
        uniform sampler2D samplerTex;
        void main()
        {
             tex = texture(samplerTex, outTexCoords);

        }
        """
    FRAGMENT_SHADER_COLOR = """
        #version 330
        in vec3 newColor;

        out vec4 g_FragColor;
        void main()
        {
             g_FragColor = vec4(newColor, 1.0f);

        }
        """

    def __init__(self, parent=None, versionprofile=None):
        """Initialize OpenGL version profile."""
        super(EditContourOpenGLWidget, self).__init__(parent)
        self.temp_directory = None
        self.versionprofile = versionprofile
        # self.image_path = image_path

        self.VAOs_list = []
        self.lines_vao_ref = []
        self.shader_program = None

        self.current_pos = QPoint()
        self.last_pos = QPoint()
        self.positions_list = []
        self.positions_list_scrolling = []

        self.main_contour_points = []  # unmapped contour points
        self.selected_point = []
        self.selected_points_for_removing = []  # used to remove points in between
        self.mapped_main_cnt = []
        # self.mapped_main_cnt_unedited = []
        self.landmark_points = []
        self.mapped_landmarks = []
        self.mapped_suggested_landmarks = []
        self.mapped_all_landmarks = []  # for showing all landmarks

        self.edit_contour_add_border = 20  # for having more place to click around contour
        # ---- tweak contour points -------------
        self.unedited_contour_points_color = [0.0, 0.0, 1.0]
        self.unedited_contour_points_size = 12.0

        self.contour_points_color = [1.0, 0.0, 0.0]
        self.contour_points_size = 10.0

        self.suggested_landmarks_color = [0.0, 0.5, 1.0]
        self.suggested_landmarks_size = 20.0

        self.all_landmarks_color = [0.0, 0.5, 1.0]
        self.all_landmarks_size = 20.0

        self.landmarks_color = [0.0, 0.0, 1.0]
        self.landmarks_size = 12.0

        self.selected_point_color = [0.0, 1.0, 0.0]
        self.selected_point_size = 11.0
        # ----------------------------------------
        self.align_to_center = False
        self.mapped_main_cnt_center = None
        self.center = [[0.0, 0.0]]
        self.project_all_contours = False
        self.mapped_multi_cnts_list = []

        self.mouse_x = None
        self.mouse_y = None

        self.image_path = None
        self.main_image = None

        self.model_view_projection_matrix = None
        self.straight_line_model_view_projection_matrix = None

        # # model matrix is also identity
        # self.model_matrix = pyrr.Matrix44.identity()
        # # initialize view matrix
        # self.view_matrix = pyrr.Matrix44.identity()

        # projection matrix is identity in our 2D case
        self.projection_matrix = pyrr.Matrix44.identity()

        self.scale_matrix = pyrr.Matrix44.identity()
        self.translate_matrix = pyrr.Matrix44.identity()

        self.current_editing_tab_index = 0
        self.editing_mode = None
        self.redefine_landmarks = False
        self.show_suggested_landmarks = False
        self.show_all_landmarks = False
        self.keep_unedited_cnt = False

        self.total_number_of_landmarks = None

        self.hand_cursor = False

        self.data_dict_list = []
        # Dictionary of main contour which should be remain undedited
        # {'mapped_cnt': ,
        #  'texture_width': ...,
        #  'texture_height': ...,
        #  'mapped_cnt_center': ...,
        # }
        self.main_contour_info_dict = {}
        # list of detected contours in dictionary format
        self.contours_info_dict_list = []
        # reset the contour to the default position
        self.reset_positions = False
        # list of resampled contours in dictionary format
        self.resampled_contours_info_dict_list = []
        # results of the procrustes analysis and generalized
        # sample : [{
        #     'mapped_cnt': ...,
        #     'texture_width': ...,
        #     'texture_height': ...,
        #     'mapped_cnt_center': ...}]
        self.procrustes_result_dict_list = []
        # sample:
        # mean_shape_dict = {'mean_shape_cnt': mean_shape_cnt,
        #                    'mean_shape_width': mean_shape_width,
        #                    'mean_shape_height': mean_shape_height,
        #                    'mean_shape_center': mean_shape_center
        #                    }
        self.generalized_proc_mean_shape_dict = {}

        self.procrustes = False
        self.generalized_procrustes = False

    def initializeGL(self):
        # print(self.width(), self.height()) # <-good to see but I will
        # delete it
        # In order to find the viewport we draw a white rectangle
        # in whole screen
        # self.create_bounding_rect()
        self.create_mvp_with_identity()
        # self.create_rectangle_with_texture(
        #     './data/ExampleLeafShapes/A alpina1.jpg')

    def clear_screen(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.main_image = None

        self.VAOs_list = []
        self.contours_info_dict_list = []
        self.selected_point = []
        self.procrustes = False
        self.generalized_procrustes = False

        self.update_opengl_widget()

    def paintGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        try:
            for data in self.VAOs_list:
                glUseProgram(data['shader'])
                VAO = data['VAO']
                object_type = data['type']
                mvp_matrix = data['mvp_matrix']

                # check if the Vertex Array Object exist
                if glIsVertexArray(VAO) == 0:
                    self.update_opengl_widget()
                    break

                glBindVertexArray(VAO)
                # draw triangle
                if object_type == GL_TRIANGLES:

                    transform_loc = glGetUniformLocation(data['shader'],
                                                         "mvp_matrix")
                    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, mvp_matrix)

                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                elif object_type == GL_LINE_STRIP:

                    glUseProgram(data['shader'])
                    transform_loc = glGetUniformLocation(data['shader'],
                                                         "mvp_matrix")
                    glUniformMatrix4fv(transform_loc, 1, GL_FALSE,
                                       mvp_matrix)
                    # glDrawElements(GL_LINE_STRIP, 2, GL_UNSIGNED_INT, None)
                    glDrawArrays(GL_LINE_STRIP, 0, data['number_of_points'])
                elif object_type == GL_POINTS:
                    glPointSize(data['point_size'])

                    # glEnable(GL_BLEND)
                    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

                    glUseProgram(data['shader'])
                    transform_loc = glGetUniformLocation(data['shader'],
                                                         "mvp_matrix")
                    glUniformMatrix4fv(transform_loc, 1, GL_FALSE,
                                       mvp_matrix)
                    # print(data['number_of_points'])
                    glDrawArrays(GL_POINTS, 0, data['number_of_points'])
        except:
            tb = traceback.format_exc()
            print(tb)
            pass
        finally:
            glBindVertexArray(0)
            glUseProgram(0)

    def resizeGL(self, width, height):
        self.update_opengl_widget()

    def update_opengl_widget(self):
        """
        This is an important function! every time we need to edit or update
        the view, first we clean the VAO list and the we create the objects.
        :return:
        """
        self.VAOs_list = []
        # self.create_bounding_rect()
        if self.generalized_procrustes:
            self.create_multiple_contours_points(
                self.procrustes_result_dict_list, 7.0)

            self.create_mean_shape_for_gprocrustes(
                self.generalized_proc_mean_shape_dict, [0.0, 0.0, 0.0], 12.0)

        if self.procrustes:
            self.create_multiple_contours_points(
                self.procrustes_result_dict_list, 7.0)

        if self.project_all_contours:
            self.create_multiple_contours_points(
                self.contours_info_dict_list, 2.0)

        if self.keep_unedited_cnt:
            try:
                self.create_unedited_contour()
            except KeyError as e:
                print("KeyError:", e)
                return

        if self.main_image and \
                (not self.generalized_procrustes and not self.procrustes and
                     not self.project_all_contours):
            self.create_main_contour_points(self.mapped_main_cnt,
                                            self.contour_points_color,
                                            self.contour_points_size)

            self.create_line_between_contour_points(self.mapped_main_cnt,
                                                    self.contour_points_color)

        try:
            if len(self.selected_point) > 0:
                self.create_heighlight_points()
        except Exception as e:
            # print('Error:', e)
            pass
        # draw landmarks if we are in editing landmark points mode
        if self.current_editing_tab_index == 0:
            if self.editing_mode == 2 and len(self.mapped_landmarks) > 0:
                self.create_landmarks(self.mapped_landmarks, self.landmarks_color,
                                      self.landmarks_size)

            if self.editing_mode == 2 and self.show_suggested_landmarks:
                self.create_landmarks(self.mapped_suggested_landmarks,
                                      self.suggested_landmarks_color,
                                      self.suggested_landmarks_size)

            if self.editing_mode == 2 and self.show_all_landmarks:
                self.create_landmarks(self.mapped_all_landmarks,
                                      self.all_landmarks_color,
                                      self.all_landmarks_size)

        self.update()
        self.reset_positions = False

    def create_object(self, object_matrix=None, indices=None,
                      gldraw_element_type=None, image_path=None,
                      use_texture=False, point_size=1.0):
        """
        create the object.
        :param object_matrix:
        :param indices:
        :param gldraw_element_type:
        :param image_path:
        :param use_texture:
        :param point_size:
        :return: python dictionary {'VAO': ..., 'type': ..., 'shader': ... ,
                                    'texture': ..., 'point_size': ...}
        """
        texture = None
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        if use_texture:
            self.shader_program = OpenGL.GL.shaders.compileProgram(
                OpenGL.GL.shaders.compileShader(self.VERTEX_SHADER,
                                                GL_VERTEX_SHADER),
                OpenGL.GL.shaders.compileShader(self.FRAGMENT_SHADER_TEX,
                                                GL_FRAGMENT_SHADER))
        else:
            self.shader_program = OpenGL.GL.shaders.compileProgram(
                OpenGL.GL.shaders.compileShader(self.VERTEX_SHADER,
                                                GL_VERTEX_SHADER),
                OpenGL.GL.shaders.compileShader(self.FRAGMENT_SHADER_COLOR,
                                                GL_FRAGMENT_SHADER))
        glUseProgram(self.shader_program)
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(object_matrix), object_matrix,
                     GL_STATIC_DRAW)
        if indices is not None:
            EBO = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * len(indices), indices,
                         GL_STATIC_DRAW)

        position = 0  # glGetAttribLocation(self.shader, 'position')
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32,
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = 1  # glGetAttribLocation(self.shader, 'color')
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 32,
                              ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        if use_texture:
            texture_coords = 2
            glVertexAttribPointer(texture_coords, 2, GL_FLOAT, GL_FALSE, 32,
                                  ctypes.c_void_p(24))
            glEnableVertexAttribArray(texture_coords)

            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

            # texture wrapping
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            # texture filtering
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            image_path = image_path
            if image_path:
                try:
                    image = Image.open(image_path)
                    img_data = image.convert("RGBA").tobytes()
                except:
                    pass

                try:
                    pixmap = QPixmap(image_path)
                    pixmap.save("{}/temp.tif".format(self.temp_directory))

                    image = Image.open("{}/temp.tif".format(self.temp_directory))
                    img_data = image.convert("RGBA").tobytes()
                except Exception as e:
                    print("can not open the image!", e)

            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width,
                         image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         img_data)

        return {'VAO': VAO, 'type': gldraw_element_type, 'shader':
            self.shader_program, 'texture': texture, 'point_size':
                    point_size}

    def create_bounding_rect(self):
        #                positions        colors         texture coords
        object_matrix = [-1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                         # Top-left
                         1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                         # Top-right
                         1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         # Bottom-right
                         -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                         # Bottom-left
                         ]
        object_matrix = np.array(object_matrix, dtype=np.float32)

        indices = [0, 1, 2,
                   2, 3, 0]
        indices = np.array(indices, dtype=np.int32)

        data_dict = self.create_object(object_matrix, indices, GL_TRIANGLES)

        # creating the view matrix
        view_matrix = pyrr.Matrix44.identity()

        # create the model-view matrix
        model_matrix = pyrr.Matrix44.identity()
        model_view_matrix = pyrr.matrix44.multiply(model_matrix, view_matrix)

        # create model view project matrix
        # projection matrix is identity in our 2D case
        projection_matrix = pyrr.Matrix44.identity()
        model_view_projection_matrix = pyrr.matrix44.multiply(
            model_view_matrix, projection_matrix)

        data_dict['mvp_matrix'] = model_view_projection_matrix

        self.VAOs_list.append(data_dict)
        self.update()

    def create_unedited_contour(self):
        """
        Create (draw) original contour under the editable contour in order to
        check the quality of the sampled or edited contour.
        :return: -
        """

        self.create_main_contour_points(self.main_contour_info_dict[
                                            'mapped_cnt'],
                                        self.unedited_contour_points_color,
                                        self.unedited_contour_points_size)

    def prepare_main_contour_points(self, cnt_points, texture_width,
                                    texture_height, image_path):
        """
        Prepare the main (current) contour to draw.
        :param cnt_points: contour points which returned from the OpenCV,
        findContour function. This function will map contour points and
        center point of the contour to the OpenGL coordinates and it will
        create a global dictionary which contains: {'mapped_cnt': ...,
        'texture_width': ..., 'texture_height': ..., 'mapped_cnt_center':...}

        :param texture_width: Width of the Texture (image)
        :param texture_height: Height of the Texture (image)
        :return: -
        """

        texture_height += self.edit_contour_add_border
        texture_width += self.edit_contour_add_border
        self.main_contour_points = cnt_points
        self.mapped_main_cnt = self.map_from_image_to_opengl(texture_width,
                                                             texture_height,
                                                             cnt_points)

        contour_center_point = Helper.compute_centroid(cnt_points)

        self.mapped_main_cnt_center = self.map_from_image_to_opengl(
            texture_width, texture_height, contour_center_point)

        self.main_contour_info_dict = {'mapped_cnt': self.mapped_main_cnt,
                                       'texture_width': texture_width,
                                       'texture_height': texture_height,
                                       'mapped_cnt_center':
                                           self.mapped_main_cnt_center,
                                       'image_path': image_path
                                       }

        if self.editing_mode == 2 and self.show_all_landmarks:
            self.prepare_all_landmarks()
        elif self.editing_mode == 2 and self.show_suggested_landmarks:
            self.prepare_suggested_landmarks()
        elif self.editing_mode == 2 and not self.show_all_landmarks:
            self.mapped_all_landmarks = []
        elif self.editing_mode == 2 and not self.show_suggested_landmarks:
            self.mapped_landmarks = []

    def prepare_resampeled_main_contour(self, cnt_points, texture_width,
                                        texture_height):
        """
        This function map the contour resampled points to the OpenGL coordinate
        :param cnt_points: contour points from OpenCV findContours function.
        :param texture_width: Width of the Texture (image)
        :param texture_height: Height of the Texture (image)
        :return: -
        """
        texture_width += self.edit_contour_add_border
        texture_height += self.edit_contour_add_border
        self.mapped_main_cnt = self.map_from_image_to_opengl(
            texture_width, texture_height, cnt_points)

    def create_main_contour_points(self, mapped_cnt_points, color_rgb,
                                   points_size):
        for points in mapped_cnt_points:
            object_matrix = self.create_object_matrix(points, color_rgb)

            object_matrix = np.array(object_matrix, dtype=np.float32)

            data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                           point_size=points_size)

            model_view_matrix = self.create_model_view_matrix()

            data_dict['mvp_matrix'] = model_view_matrix

            data_dict['number_of_points'] = int(len(object_matrix) / 8)

            self.VAOs_list.append(data_dict)

    def create_model_view_matrix(self):
        """
        create model view matrix for the current image (contour)
        :return: 4x4 mvp_matrix
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[2] = self.width()
        view_port[3] = self.height()

        texture_width = self.main_contour_info_dict['texture_width']
        texture_height = self.main_contour_info_dict['texture_height']
        # contour_points = self.main_contour_info_dict['mapped_cnt']
        cnt_center_point = self.main_contour_info_dict['mapped_cnt_center'][
            0][0]

        image_ratio = texture_width / texture_height
        screen_ratio = view_port[2] / view_port[3]

        # scale the image according the ratio
        # creating the view matrix
        view_matrix = pyrr.Matrix44().identity()

        if image_ratio > screen_ratio:
            view_matrix[0][0] = 1
            view_matrix[1][1] = 1 * screen_ratio / image_ratio
        else:
            view_matrix[0][0] = 1 / screen_ratio * image_ratio
            view_matrix[1][1] = 1
        view_matrix[2][2] = -1
        view_matrix[3][3] = 1
        view_matrix[0][3] = 0
        view_matrix[1][3] = 0
        view_matrix[2][3] = 0

        # create the model-view matrix
        model_matrix = pyrr.Matrix44.identity()

        model_view_matrix = pyrr.matrix44.multiply(model_matrix,
                                                   view_matrix)

        # create model view project matrix
        # projection matrix is identity in our 2D case
        projection_matrix = pyrr.Matrix44.identity()
        model_view_projection_matrix = pyrr.matrix44.multiply(
            model_view_matrix, projection_matrix)

        if self.reset_positions:
            self.scale_matrix = pyrr.Matrix44.identity()
            self.translate_matrix = pyrr.Matrix44.identity()
            return model_view_matrix

        if self.align_to_center:  # and contour_points is not None:
            translate_matrix = pyrr.matrix44.create_from_translation(
                [self.center[0][0] - cnt_center_point[0][0], self.center[
                    0][1] - cnt_center_point[0][1], 0, 0])

            translate_to_center_mvp_matrix = pyrr.matrix44.multiply(
                translate_matrix, model_view_matrix)

            model_view_matrix = translate_to_center_mvp_matrix

        model_view_matrix = pyrr.matrix44.multiply(self.scale_matrix,
                                                   model_view_matrix)

        model_view_matrix = pyrr.matrix44.multiply(self.translate_matrix,
                                                   model_view_matrix)

        return model_view_matrix

    def create_main_contour_center_point(self, mapped_cnt_center_point,
                                         color_rgb, points_size):
        object_matrix = self.create_object_matrix(mapped_cnt_center_point[0],
                                                  color_rgb)
        object_matrix = np.array(object_matrix, dtype=np.float32)

        data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                       point_size=points_size)

        model_view_matrix = self.create_model_view_matrix()

        data_dict['mvp_matrix'] = model_view_matrix

        data_dict['number_of_points'] = int(len(object_matrix) / 8)

        self.VAOs_list.append(data_dict)

    def create_line_between_contour_points(self, mapped_cnt_points,
                                           color_rgb, model_view_matrix=None):
        """
        This function draw line between contour points.
        :param mapped_cnt_points: contour points which mapped to the OpenGL
        coordinates.
        :param color_rgb: vector (list) of RGB color. e.x:[1.0, 1.0, 0.0]
        :return: -
        """
        for points in mapped_cnt_points:
            object_matrix = self.create_line_object_matrix(points,
                                                           color_rgb)

            object_matrix = np.array(object_matrix, dtype=np.float32)

            data_dict = self.create_object(object_matrix, None, GL_LINE_STRIP)
            if model_view_matrix is not None:
                mv_matrix = model_view_matrix
            else:
                mv_matrix = self.create_model_view_matrix()

            data_dict['mvp_matrix'] = mv_matrix
            data_dict['number_of_points'] = int(len(object_matrix) / 8)

            self.VAOs_list.append(data_dict)

    def create_center_point_of_the_opengl(self):
        color_rgb = [0.0, 0.0, 0.0]
        object_matrix = self.create_object_matrix([[[0.0, 0.0, -1.0]]],
                                                  color_rgb)

        object_matrix = np.array(object_matrix, dtype=np.float32)

        data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                       point_size=10.0)

        # model_view_matrix = self.create_model_view_matrix()

        data_dict['mvp_matrix'] = pyrr.Matrix44.identity()

        data_dict['number_of_points'] = int(len(object_matrix) / 8)

        self.VAOs_list.append(data_dict)

    def create_heighlight_points(self):
        object_matrix = self.create_object_matrix_selected_point(
            self.selected_point, self.selected_point_color)

        object_matrix = np.array(object_matrix, dtype=np.float32)

        data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                       point_size=self.selected_point_size)

        model_view_matrix = self.create_model_view_matrix()

        data_dict['mvp_matrix'] = model_view_matrix
        data_dict['number_of_points'] = int(len(object_matrix) / 8)

        self.VAOs_list.append(data_dict)

    def prepare_multiple_contours_points(self, data_dict_list):
        """
        prepare contours in order to draw.
        This function map all the contours and center points to OpenGL
        coordinates base on the width and height of the image (texture).
        This function also create a global list of python dictionaries in
        order to use them when drawing the textures.

        :param data_dict_list: list of python dictionaries.
        The keys of each dictionary are: {'contour': ...,
                                          'image_path': ...,
                                          'current_image': ...}
        :return: -
        """
        self.contours_info_dict_list = []
        for data in data_dict_list:
            if not data['contour']:
                continue

            texture_height, texture_width = Helper.get_image_height_width(data['image_path'])
            if data['image_path'] == self.main_contour_info_dict['image_path']:
                continue
            # if not data.get('texture_width', None):
            #     image = Image.open(data['image_path'])
            #     texture_width = image.width
            #     texture_height = image.height
            # else:
            #     texture_width = data.get('texture_width')
            #     texture_height = data.get('texture_height')
            mapped_cnt = self.map_from_image_to_opengl(
                texture_width, texture_height, data['contour'])

            contour_center_point = ProcrustesAnalysisWorkerThread.compute_centroid(
                data['contour'])
            mapped_cnt_center = self.map_from_image_to_opengl(
                texture_width, texture_height, contour_center_point)

            self.contours_info_dict_list.append({
                'mapped_cnt': mapped_cnt,
                'texture_width': texture_width,
                'texture_height': texture_height,
                'mapped_cnt_center': mapped_cnt_center,
                'color_rgb': data['color_rgb']
            })
        #print("len(self.contours_info_dict_list)=", len(self.contours_info_dict_list))

    def create_multiple_contours_points(self, contours_info_dict_list,
                                        points_size):
        """
        when user select the show all contours this function will create
        each of the contours on the edit contour widget. This function will
        use the list of dictionary that prepare function created before.

        :param contours_info_dict_list:
        :param points_size: size of the contour points
        :return: -
        """
        if contours_info_dict_list is None:
            return

        for data in contours_info_dict_list:

            mapped_cnt = data.get('mapped_resampled_cnt', None)

            if mapped_cnt is None:
                mapped_cnt = data.get('mapped_cnt', None)
                if mapped_cnt is None:
                    continue
            # print(data['result_image_path'], mapped_cnt)
            texture_width = data['texture_width']
            texture_height = data['texture_height']
            mapped_cnt_center = Helper.compute_centroid(mapped_cnt)#data['mapped_cnt_center'][0][0]
            mapped_cnt_center = self.map_from_image_to_opengl(texture_width,
                                                             texture_height,
                                                             mapped_cnt_center)
            color_rgb = data['color_rgb']

            model_view_matrix = self.create_multiple_model_view_matrix(
                texture_width, texture_height, mapped_cnt_center)

            mapped_cnt = self.map_from_image_to_opengl(
                    texture_width, texture_height, mapped_cnt)

            for points in mapped_cnt:
                object_matrix = self.create_object_matrix(points, color_rgb)

                object_matrix = np.array(object_matrix, dtype=np.float32)

                data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                               point_size=points_size)

                data_dict['mvp_matrix'] = model_view_matrix

                data_dict['number_of_points'] = int(len(object_matrix) / 8)

                self.VAOs_list.append(data_dict)

                # draw line between points
                self.create_line_between_contour_points(mapped_cnt, color_rgb,
                                                        model_view_matrix)

    def create_multiple_model_view_matrix(self, texture_width, texture_height,
                                          cnt_center_point):
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[2] = self.width()
        view_port[3] = self.height()

        image_ratio = texture_width / texture_height
        screen_ratio = view_port[2] / view_port[3]

        # scale the image according the ratio
        # creating the view matrix
        view_matrix = pyrr.Matrix44().identity()

        if image_ratio > screen_ratio:
            view_matrix[0][0] = 1
            view_matrix[1][1] = 1 * screen_ratio / image_ratio
        else:
            view_matrix[0][0] = 1 / screen_ratio * image_ratio
            view_matrix[1][1] = 1
        view_matrix[2][2] = -1
        view_matrix[3][3] = 1
        view_matrix[0][3] = 0
        view_matrix[1][3] = 0
        view_matrix[2][3] = 0

        # create the model-view matrix
        model_matrix = pyrr.Matrix44.identity()

        model_view_matrix = pyrr.matrix44.multiply(model_matrix,
                                                   view_matrix)

        # create model view project matrix
        # projection matrix is identity in our 2D case
        projection_matrix = pyrr.Matrix44.identity()
        model_view_projection_matrix = pyrr.matrix44.multiply(
            model_view_matrix, projection_matrix)

        if self.reset_positions and not self.align_to_center:
            self.scale_matrix = pyrr.Matrix44.identity()
            self.translate_matrix = pyrr.Matrix44.identity()
            return model_view_matrix

        if self.align_to_center:  # and contour_points is not None:
#            print(self.center)
#            print(self.center[0][0])
#            print(self.center[0][1])
#            print(cnt_center_point[0][0][0][0])
#            print(cnt_center_point[0][0][0][1])
#            print(cnt_center_point)
            translate_matrix = pyrr.matrix44.create_from_translation(
                [self.center[0][0] - cnt_center_point[0][0][0][0], self.center
                    [0][1] - cnt_center_point[0][0][0][1], 0, 0])

            translate_to_center_mvp_matrix = pyrr.matrix44.multiply(
                translate_matrix, model_view_matrix)

            model_view_matrix = translate_to_center_mvp_matrix

        model_view_matrix = pyrr.matrix44.multiply(self.scale_matrix,
                                                   model_view_matrix)

        model_view_matrix = pyrr.matrix44.multiply(self.translate_matrix,
                                                   model_view_matrix)

        return model_view_matrix

    def create_mean_shape_for_gprocrustes(self, mean_shape_dict, color_rgb,
                                          points_size):

        # color_rgb = [random.uniform(0, 1), random.uniform(0, 1),
        #              random.uniform(0, 1)]
        if mean_shape_dict is None:
            return
        mapped_cnt = mean_shape_dict['mean_shape_cnt']
        texture_width = mean_shape_dict['mean_shape_width']
        texture_height = mean_shape_dict['mean_shape_height']
        mapped_cnt_center = mean_shape_dict['mean_shape_center'][0][0]

        model_view_matrix = self.create_multiple_model_view_matrix(
            texture_width, texture_height, mapped_cnt_center)

        for points in mapped_cnt:
            object_matrix = self.create_object_matrix(points, color_rgb)

            object_matrix = np.array(object_matrix, dtype=np.float32)

            data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                           point_size=points_size)

            data_dict['mvp_matrix'] = model_view_matrix

            data_dict['number_of_points'] = int(len(object_matrix) / 8)

            self.VAOs_list.append(data_dict)

            # draw line between points
            self.create_line_between_contour_points(mapped_cnt, color_rgb,
                                                    model_view_matrix)

    @to_contour_opencvformat
    def add_contour_point(self, input_point):
        temp_list = []
        first, second = Helper.find_closest_line_segment(
            self.mapped_main_cnt, input_point)
        # first, second = Helper.find_two_nearest_points(self.mapped_main_cnt,
        #                                                input_point)

        input_point_3d = np.array([[input_point[0], input_point[1], -1.0]])
        for cnt in self.mapped_main_cnt:
            for point in cnt:
                if Helper.compare_equality(first, point) and not \
                        Helper.contain_list_or_array(input_point_3d, temp_list):
                    temp_list.append(point)
                    temp_list.append(input_point_3d)
                elif Helper.compare_equality(second, point) and not \
                        Helper.contain_list_or_array(input_point_3d, temp_list):
                    temp_list.append(input_point_3d)
                    temp_list.append(point)
                else:
                    temp_list.append(point)

        return temp_list

    def prepare_suggested_landmarks(self):
        """
        find contour and detect the landmarks automatically.
        :return: None
        """

        texture_height, texture_width = Helper.get_image_height_width(self.image_path)
        texture_height += self.edit_contour_add_border
        texture_width += self.edit_contour_add_border

        landmarks = Helper.find_landmarks(
            self.main_contour_points, texture_width, texture_height)

        # after mapping, the position of the top and bottom will be change
        # because of the different coordinate systems
        mapped_landmarks = self.map_from_image_to_opengl(texture_width,
                                                         texture_height, landmarks)

        self.mapped_suggested_landmarks = mapped_landmarks

    def prepare_all_landmarks(self):
        """
        find contour and detect the landmarks automatically.
        :return: None
        """

        texture_height, texture_width = Helper.get_image_height_width(self.image_path)
        texture_height += self.edit_contour_add_border
        texture_width += self.edit_contour_add_border

        landmarks = Helper.find_landmarks(
            self.main_contour_points, texture_width, texture_height,
            self.temp_directory, self.image_path)

        # after mapping, the position of the top and bottom will be change
        # because of the different coordinate systems
        mapped_landmarks = self.map_from_image_to_opengl(texture_width,
                                                         texture_height, landmarks)

        self.mapped_all_landmarks = mapped_landmarks

    def create_landmarks(self, landmarks, color_rgb, points_size):
        """
        create landmarks and add them to VAO list in order to draw them.
        :param landmarks: landmarks points
        :param color: color of the land marks
        :param points_size: size of the points
        :return:
        """

        if np.array(landmarks).shape[-1] == 3:
            landmarks = [np.array(landmarks).reshape(-1, 1, 3)]
        counter = 1
        for landmark in landmarks[0]:
            random.seed(counter)
            color_rgb = [random.uniform(0, 1), random.uniform(0, 1),
                         random.uniform(0, 1)]

            object_matrix = self.create_object_matrix([landmark], color_rgb)

            object_matrix = np.array(object_matrix, dtype=np.float32)

            data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                           point_size=points_size)

            model_view_matrix = self.create_model_view_matrix()

            data_dict['mvp_matrix'] = model_view_matrix
            data_dict['number_of_points'] = int(len(object_matrix) / 8)

            self.VAOs_list.append(data_dict)

            counter += 1

    def select_or_add_landmark(self, x, y):
        mlands_list = []
        min_dist = 1000
        for cnt in self.mapped_main_cnt:
            for data in cnt:
                dist = Helper.euclidean_distance((x, y),
                                                 (data[0][0],
                                                  data[0][1]))
                if dist < min_dist:
                    point = data.tolist()
                    min_dist = dist

        min_dist = 1000
        for landmarks in self.mapped_landmarks:
            for landmark in landmarks:
                dist = Helper.euclidean_distance(
                    (point[0][0], point[0][1]),
                    (landmark[0][0], landmark[0][1])
                )
                if dist < min_dist:
                    min_landmark = landmark
                    min_dist = dist
        # If the land mark is not exist in the mapped_landmarks list (in
        #  the case of not showing suggested landmarks) we only add the point
        #  to the list
        if self.redefine_landmarks:
            try:
                for landmarks in self.mapped_landmarks:
                    if len(landmarks) == self.total_number_of_landmarks:
                        landmarks = landmarks.tolist()
                        landmarks.remove(min_landmark.tolist())
                        mlands_list = landmarks
                    else:
                        mlands_list = landmarks

                if type(mlands_list) is list:
                    mlands_list.append(point)
                else:
                    mlands_list = mlands_list.tolist()
                    mlands_list.append(point)

                self.mapped_landmarks = np.asarray([mlands_list])
            except Exception as e:
                print(e)
                self.mapped_landmarks = self.mapped_landmarks.tolist()
                self.mapped_landmarks.append(np.asarray(point, dtype=np.float32))

        self.update_opengl_widget()

    def mousePressEvent(self, event):
        try:
            self.mouse_x = event.pos().x()
            self.mouse_y = event.pos().y()
            if not self.is_point_on_image(self.mouse_x, self.mouse_y):
                print("Please click closer to the contours!")
                return
            x_new, y_new, z_new = self.map_qt_to_opengl_coordinates(
                self.mouse_x, self.mouse_y)
        except:
            return
        if QGuiApplication.keyboardModifiers() == Qt.AltModifier:
            self.setCursor(Qt.ClosedHandCursor)

        elif self.editing_mode == 1:
            # editing the contour points
            if QGuiApplication.keyboardModifiers() == Qt.ControlModifier:
                self.mapped_main_cnt = self.add_contour_point([x_new, y_new])
                self.update_opengl_widget()

            # use Shift for removing part of the contour
            elif QGuiApplication.keyboardModifiers() == Qt.ShiftModifier:
                width = self.main_contour_info_dict['texture_width']
                height = self.main_contour_info_dict['texture_height']
                self.selected_point = Helper.nearest_point(
                    self.mapped_main_cnt, (x_new, y_new))
                if len(self.selected_points_for_removing) > 3:
                    self.selected_points_for_removing.pop()

                mapped_point = Helper.map_from_opengl_to_image(
                    (x_new, y_new, -1),
                    glGetIntegerv(GL_VIEWPORT), width, height)
                self.selected_points_for_removing.append(
                    mapped_point)
                if len(self.selected_points_for_removing) == 2:
                    edited_contour = Helper.remove_contours_between_two_selected(
                        self.temp_directory,
                        self.selected_points_for_removing,
                        self.main_contour_info_dict['image_path'])

                    self.mapped_main_cnt = self.map_from_image_to_opengl(
                        width, height, edited_contour)

                    self.selected_points_for_removing = []
                self.update_opengl_widget()
            else:
                self.selected_point = Helper.nearest_point(
                    self.mapped_main_cnt, (x_new, y_new))
                self.update_opengl_widget()

        elif self.editing_mode == 2:
            mapped_point = Helper.nearest_point(
                self.mapped_main_cnt, (x_new, y_new))

            # editing the landmarks (add or select)
            if QGuiApplication.keyboardModifiers() == Qt.ControlModifier:
                self.select_or_add_landmark(mapped_point[0][0], mapped_point[0][1])

                # if QGuiApplication.keyboardModifiers() == Qt.AltModifier:
                #     if len(self.positions_list_scrolling) == 0:
                #         self.positions_list_scrolling.append((x_new, y_new))

    def landmarks_map_back_to_image(self):

        landmarks = np.vstack(self.mapped_landmarks).squeeze()
        texture_width = self.main_contour_info_dict['texture_width']
        texture_height = self.main_contour_info_dict['texture_height']
        view_port = glGetIntegerv(GL_VIEWPORT)
        # print(texture_width, texture_height, view_port)

        points = []
        for landmark in landmarks:
            mapped_point = Helper.map_from_opengl_to_image(
                landmark, view_port,
                texture_width, texture_height)
            # mapped_point = Helper.change_number_of_contour_coordinates(
            #     mapped_point, remove_axis='z')
            points.append(mapped_point)

        # self.contours_info_dict_list

        return np.asarray(points)

    def mouseMoveEvent(self, event):
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()
        try:
            x_new, y_new, z_new = self.map_qt_to_opengl_coordinates(mouse_x,
                                                                    mouse_y
                                                                    )
        except:
            return
        if QGuiApplication.keyboardModifiers() == Qt.AltModifier:
            self.positions_list_scrolling.append((x_new, y_new))
            if len(self.positions_list_scrolling) >= 2:
                self.scroll_image()

    def mouseReleaseEvent(self, event):
        self.positions_list_scrolling[:] = []
        self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, QKeyEvent):
        # Delete the point with Backspace key on the keyboard
        if QKeyEvent.key() == Qt.Key_Backspace:
            try:
                changed_map = []
                for i in range(len(self.mapped_main_cnt)):
                    cnt = self.mapped_main_cnt[i]
                    cnt = cnt.tolist()
                    for j, data in enumerate(cnt):
                        if (data[0] == self.selected_point[0]).all():
                            cnt.pop(j)
                            break

                    changed_map.append(np.asarray(cnt))
                    self.mapped_main_cnt = changed_map

                # if self.selected_point not in self.mapped_main_cnt:
                self.selected_point = []

                self.update_opengl_widget()
                # print("length=", len(self.mapped_main_cnt))
            except Exception as e:
                print('Error:', e)
                pass

        if not self.hand_cursor and QKeyEvent.key() == Qt.Key_Alt:
            self.setCursor(Qt.OpenHandCursor)
            self.hand_cursor = True
        else:
            self.setCursor(Qt.ArrowCursor)
            self.hand_cursor = False
        if QKeyEvent.key() == Qt.Key_Escape:
            if self.editing_mode == 1:
                self.selected_point = None
                self.update_opengl_widget()
            elif self.editing_mode == 2:
                if len(self.mapped_landmarks) > 0:
                    self.mapped_landmarks.pop()
                    self.update_opengl_widget()

    def wheelEvent(self, qwheelevent):
        if qwheelevent.angleDelta().y() > 0:
            self.zoom_in_opengl_widget(1.5)
        if qwheelevent.angleDelta().y() < 0:
            self.zoom_in_opengl_widget(0.75)

    def is_point_on_image(self, mouse_x, mouse_y):
        """
        check if the mouse click is happened on the image.
        :param mouse_x:
        :param mouse_y:
        :return:
        """
        x, y, z = self.map_qt_to_opengl_coordinates(mouse_x, mouse_y, 0)
        # print(x, y)
        if -1 <= x <= 1 and -1 <= y <= 1:
            return True

    def draw_line(self, last_pos, current_pos):
        """
        create line object. it will draw line between two points.
        :param last_pos:
        :param current_pos:
        :return:
        """
        object_matrix = [last_pos[0], last_pos[1], 0.0, 1.0, 0.0, 0.0, 0.0,
                         0.0,
                         current_pos[0], current_pos[1], 0.0, 1.0, 0.0, 0.0,
                         1.0, 0.0,
                         ]
        object_matrix = np.array(object_matrix, dtype=np.float32)
        indices = [0, 1]
        indices = np.array(indices, dtype=np.int32)
        data_dict = self.create_object(object_matrix, indices,
                                       GL_TRIANGLES, use_texture=False)

        # self.prepare_matrices_for_draw_line(data_dict)
        # update
        self.update_opengl_widget()

    def draw_line_with_width(self, last_pos, current_pos, line_width):
        """
        This function will draw a line with the specified width. This
        function actually draw a rectangle and fill it with a color.
        :param last_pos:
        :param current_pos:
        :param line_width:
        :return:
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[3] = self.height()
        view_port[2] = self.width()
        ratio = view_port[2] / view_port[3]

        # use image model view in order to draw only on image
        self.model_view_projection_matrix = pyrr.matrix44.multiply(
            self.model_view_matrix, self.projection_matrix)
        # draw line if the control key (command key) is pressed
        if QGuiApplication.keyboardModifiers() == Qt.ControlModifier:
            alpha = self.angle(last_pos, current_pos)
            # model_matrix[1][1] = math.cos(alpha)
            # model_matrix[1][2] = -1 * math.sin(alpha)
            # model_matrix[2][1] = math.sin(alpha)
            # model_matrix[2][2] = math.cos(alpha)
            #
            # self.model_view_matrix = pyrr.matrix44.multiply(model_matrix,
            #                                                 view_matrix)

            self.draw_line(last_pos, current_pos)

        # Horizontal line
        elif abs(last_pos[0] - current_pos[0]) > abs(last_pos[1] -
                                                             current_pos[1]):
            if ratio > 1:
                line_width *= ratio
            else:
                line_width /= ratio

            line_y = last_pos[1]
            x0 = (last_pos[0], line_y - line_width / 2)  # bottom left
            x1 = (current_pos[0], line_y - line_width / 2)  # bottom right
            x2 = (current_pos[0], line_y + line_width / 2)  # top right
            x3 = (last_pos[0], line_y + line_width / 2)  # top left

            object_matrix = [x3[0], x3[1], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                             # Top-left
                             x2[0], x2[1], 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                             # Top-right
                             x1[0], x1[1], 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                             # Bottom-right
                             x0[0], x0[1], 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                             # Bottom-left
                             ]
            object_matrix = np.array(object_matrix, dtype=np.float32)
            indices = [0, 1, 2,
                       2, 3, 0]
            indices = np.array(indices, dtype=np.int32)

            data_dict = self.create_object(object_matrix, indices,
                                           GL_TRIANGLES, use_texture=False)
            # draw on image
            data_dict['mvp_matrix'] = self.model_view_projection_matrix

            self.VAOs_list.append(data_dict)

            # self.update_opengl_widget()

        # Vertical line
        elif abs(last_pos[0] - current_pos[0]) < abs(last_pos[1] -
                                                             current_pos[1]):
            if ratio < 1:
                line_width *= ratio
            else:
                line_width /= ratio
            line_x = last_pos[0]
#            print(line_x)
            x0 = (line_x - line_width / 2, last_pos[1])  # bottom left
            x1 = (line_x + line_width / 2, last_pos[1])  # bottom right
            x2 = (line_x + line_width / 2, current_pos[1])  # top right
            x3 = (line_x - line_width / 2, current_pos[1])  # top left

            object_matrix = [x3[0], x3[1], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                             # Top-left
                             x2[0], x2[1], 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                             # Top-right
                             x1[0], x1[1], 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                             # Bottom-right
                             x0[0], x0[1], 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                             # Bottom-left
                             ]

            object_matrix = np.array(object_matrix, dtype=np.float32)
            indices = [0, 1, 2,
                       2, 3, 0]
            indices = np.array(indices, dtype=np.int32)
            data_dict = self.create_object(object_matrix, indices,
                                           GL_TRIANGLES, use_texture=False)

            data_dict['mvp_matrix'] = self.model_view_projection_matrix

            self.VAOs_list.append(data_dict)

            # self.update_opengl_widget()

    def create_mvp_with_identity(self):
        # creating the view matrix
        view_matrix = pyrr.Matrix44.identity()

        # create the model-view matrix
        model_matrix = pyrr.Matrix44.identity()
        model_view_matrix = pyrr.matrix44.multiply(model_matrix,
                                                   view_matrix)

        # create model view project matrix
        # projection matrix is identity in our 2D case
        projection_matrix = pyrr.Matrix44.identity()
        model_view_projection_matrix = pyrr.matrix44.multiply(
            model_view_matrix, projection_matrix)

        self.straight_line_model_view_projection_matrix = \
            model_view_projection_matrix

    def zoom_in_opengl_widget(self, factor):
        """
        zoom in image by the given factor
        :param factor: zoom in scale factor
        :return:
        """
        s_matrix = pyrr.matrix44.create_from_scale([factor, factor,
                                                    1, 1])
        self.scale_matrix = pyrr.matrix44.multiply(self.scale_matrix, s_matrix)
        for data in self.VAOs_list:
            if data['type'] == GL_POINTS or data['type'] == GL_LINE_STRIP:
                data['mvp_matrix'] = pyrr.matrix44.multiply(s_matrix,
                                                            data['mvp_matrix'])
                data['mvp_matrix'][3][0] *= factor
                data['mvp_matrix'][3][1] *= factor

        self.update()

    def zoom_out_opengl_widget(self, factor):
        """
        zoom out image byt the given factor
        :param factor: zoom out scale factor
        :return:
        """
        s_matrix = pyrr.matrix44.create_from_scale([factor, factor,
                                                    1, 1])
        self.scale_matrix = pyrr.matrix44.multiply(self.scale_matrix, s_matrix)
        for data in self.VAOs_list:
            if data['type'] == GL_POINTS or data['type'] == GL_LINE_STRIP:
                data['mvp_matrix'] = pyrr.matrix44.multiply(s_matrix,
                                                            data['mvp_matrix'])
                data['mvp_matrix'][3][0] *= factor
                data['mvp_matrix'][3][1] *= factor

        self.update()

    def scroll_image(self):
        """
        Change position of the camera (translate) in x and y directions
        according to the mouse movement on the image in the OpenGL widget.
        :return:
        """
        x1, y1 = self.positions_list_scrolling[0]
        x2, y2 = self.positions_list_scrolling[1]
        diff_x = (x2 - x1)
        diff_y = (y2 - y1)
        translate_m = pyrr.matrix44.create_from_translation(
            [diff_x, diff_y, 0, 0])

        self.translate_matrix = pyrr.matrix44.multiply(self.translate_matrix,
                                                       translate_m)
        for data in self.VAOs_list:
            if data['type'] == GL_POINTS or data['type'] == GL_LINE_STRIP:
                data['mvp_matrix'] = pyrr.matrix44.multiply(translate_m,
                                                            data['mvp_matrix'])
            self.update()
            self.positions_list_scrolling = []

    def map_qt_to_opengl_coordinates(self, x, y, z=0):
        """
        Map point in qt widget coordinate to OpenGL homogeneous coordinate

        :param x: number in X axis
        :param y: number in Y axis
        :param z: number in Z axis
        :return: (x,y,z) in new coordinate
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        if view_port[2] == 2 * self.width():
            view_port[3] = int(view_port[3] / 2)
            view_port[2] = int(view_port[2] / 2)
        else:
            view_port[3] = self.height()
            view_port[2] = self.width()
        real_y = view_port[3] - y
        model_view_matrix = self.create_model_view_matrix()

        x_new, y_new, z_new = GLU.gluUnProject(x, real_y, z,
                                               model_view_matrix,
                                               self.projection_matrix,
                                               view_port)
        return x_new, y_new, z_new

    def map_opengl_to_qt_coordinates(self, x, y, z=0):
        """
        Map point in OpenGL homogeneous coordinate [-1,1] to qt widget
        coordinate.
        :param x: number in X axis
        :param y: number in Y axis
        :param z: number in Y axis
        :return: (x,y,z) in new coordinate
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        model_view_matrix = self.create_model_view_matrix()
        widget_x, widget_y, widget_z = GLU.gluProject(x, y, z,
                                                      model_view_matrix,
                                                      self.projection_matrix,
                                                      view_port)

        return widget_x, widget_y, widget_z

    def map_from_image_to_opengl(self, width, height, contours):
        """

        :param width: Width of the texture (image)
        :param height: Height of the texture (image)
        :param contours: contours which retrieved from OpenCV findContours
        function
        :return: numpy array e.x: [array([[[x1, y1, z1]],[[x2, y2, z2]]...])]
        """
        view_port = glGetIntegerv(GL_VIEWPORT)

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
            if len(point)==1:
                point = point[0]
            x = point[0]
            y = point[1]
            z = 0
            model_view_matrix = pyrr.Matrix44.identity()
            real_y = view_port[3] - y
            x_new, y_new, z_new = GLU.gluUnProject(x, real_y, z,
                                                   model_view_matrix,
                                                   self.projection_matrix,
                                                   view_port)

            coordinate.append(x_new)
            coordinate.append(y_new)
            coordinate.append(z_new)
        all_cnt_coordinates.append(np.array(coordinate).reshape((-1, 1,
                                                                 3)))
        return all_cnt_coordinates

    def map_from_opengl_to_image(self, width, height, contours):
        """

        :param width: Width of the texture (image)
        :param height: Height of the texture (image)
        :param contours:
        :return: numpy array e.x: [array([[[x1, y1, z1]],[[x2, y2, z2]]...])]
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[2] = width
        view_port[3] = height
        model_view_matrix = pyrr.Matrix44.identity()
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
            y = point[0][1] * -1
            z = point[0][2]
            x_new, y_new, z_new = GLU.gluProject(x, y, z,
                                                 model_view_matrix,
                                                 self.projection_matrix,
                                                 view_port)
            coordinate.append(x_new)
            coordinate.append(y_new)
            # coordinate.append(z_new)

        all_cnt_coordinates.append(np.array(coordinate).reshape((-1, 1,
                                                                 2)))

        return all_cnt_coordinates

    def create_object_matrix(self, coordinates, color_rgb):
        """
        create object matrix from the set of points
        :param coordinates: [array([[[x1,y1,z1]], [[x2,y2,z2]], ...])]
        :param color_rgb: [r, g, b]
        :return:
        """
        object_matrix = []
        for point in coordinates:
            object_matrix.append(point[0][0])
            object_matrix.append(point[0][1])
            object_matrix.append(point[0][2])
            # color
            object_matrix.append(color_rgb[0])
            object_matrix.append(color_rgb[1])
            object_matrix.append(color_rgb[2])
            # texture
            object_matrix.append(0.0)
            object_matrix.append(0.0)

        return object_matrix

    def create_object_matrix_selected_point(self, point, color):
        """
        create object matrix for the chosen point
        :param point:
        :return:
        """
        object_matrix = []
        object_matrix.append(point[0][0])
        object_matrix.append(point[0][1])
        object_matrix.append(point[0][2])
        # color
        object_matrix.append(color[0])
        object_matrix.append(color[1])
        object_matrix.append(color[2])
        # texture
        object_matrix.append(0.0)
        object_matrix.append(0.0)

        return object_matrix

    def create_line_object_matrix(self, coordinates, color_rgb):
        """
        create object matrix from the set of points
        :param coordinates: [array([[[x1,y1,z1]], [[x2,y2,z2]], ...])]
        :param color_rgb: [r, g, b]
        :return: object matrix [x1, y1, z1, x2, y2, z2, ...]
        """
        object_matrix = []
        for point in coordinates:
            object_matrix.append(point[0][0])
            object_matrix.append(point[0][1])
            object_matrix.append(0.0)
            # color
            object_matrix.append(color_rgb[0])
            object_matrix.append(color_rgb[1])
            object_matrix.append(color_rgb[2])
            # texture
            object_matrix.append(0.0)
            object_matrix.append(0.0)

        # connect last point to first one
        point = coordinates[0]
        object_matrix.append(point[0][0])
        object_matrix.append(point[0][1])
        object_matrix.append(0.0)
        # color
        object_matrix.append(color_rgb[0])
        object_matrix.append(color_rgb[1])
        object_matrix.append(color_rgb[2])
        # texture
        object_matrix.append(0.0)
        object_matrix.append(0.0)

        return object_matrix

    def prepare_resampled_contours(self, resampled_cnts_dict_list):
        self.resampled_contours_info_dict_list = []

        for data in resampled_cnts_dict_list:
#AR: This needs to be correctly handled in the "or" case
            if data.get('resampled_contour', None) is None and data.get('mapped_resampled_contour', None) is None:
                continue

            # image = Image.open(data['image_path'])
            # texture_width = image.width
            # texture_height = image.height
            if data.get('mapped_resampled_contour', None):
                texture_height, texture_width = Helper.get_image_height_width(data['image_path'])
                self.resampled_contours_info_dict_list.append({
                    'mapped_resampled_cnt': data['mapped_resampled_contour'],
                    'resampled_cnt': data['resampled_contour'],
                    'texture_width': texture_width,
                    'texture_height': texture_height,
                    'result_image_path': data['image_path'],
                    'color_rgb': data['color_rgb']
                })
            else:
                texture_height, texture_width = Helper.get_image_height_width(data['image_path'])
                mapped_resampled_cnt = self.map_from_image_to_opengl(
                    texture_width, texture_height, data['resampled_contour'])

                self.resampled_contours_info_dict_list.append({
                    'mapped_resampled_cnt': mapped_resampled_cnt,
                    'resampled_cnt': data['resampled_contour'],
                    'texture_width': texture_width,
                    'texture_height': texture_height,
                    'result_image_path': data['image_path'],
                    'color_rgb': data['color_rgb']
                })

    def save_procrustes_results(self, directory):
        if self.procrustes_result_dict_list is None:
            return
        if self.generalized_proc_mean_shape_dict is None:
            return

        for data in self.procrustes_result_dict_list:
            width = data['texture_width']
            height = data['texture_height']
            result_image_path = data['result_image_path']
            mapped_back_to_image = data['resampled_cnt']#self.map_from_opengl_to_image(
            #    width, height, data['mapped_resampled_cnt'])
            

            Helper.save_aligned_contour_csv(directory,
                                            mapped_back_to_image,
                                            result_image_path)
        try:
            width = self.generalized_proc_mean_shape_dict[
                'mean_shape_width']
            height = self.generalized_proc_mean_shape_dict[
                'mean_shape_height']
            result_image_path = "mean_shape"

            mapped_back_to_image = self.map_from_opengl_to_image(
                width, height,
                self.generalized_proc_mean_shape_dict[
                    'mean_shape_cnt'])
            # cnt = self.generalized_proc_mean_shape_dict[
            #                                     'mean_shape_cnt']
            # cnt = Helper.change_number_of_contour_coordinates(cnt)

            Helper.save_aligned_contour_csv(directory,
                                            mapped_back_to_image,
                                            result_image_path)
        except KeyError:
            pass

    def save_edited_contour_or_landmarks_results(self, temp_directory, result_image_path):
        width = self.main_contour_info_dict['texture_width']
        height = self.main_contour_info_dict['texture_height']

        resampled_points = Helper.load_resampled_from_csv(
            temp_directory, result_image_path)

        if self.editing_mode == 1:
            try:
                if len(resampled_points[0]) > len(self.mapped_main_cnt[0]):
                    mapped_back_to_image = self.map_from_opengl_to_image(
                        width, height, self.mapped_main_cnt)

                    Helper.save_resampled_csv(temp_directory,
                                              mapped_back_to_image,
                                              result_image_path)
                else:
                    # contour = Helper.load_contours_from_csv(
                    #     directory, result_image_path)
                    mapped_back_to_image = self.map_from_opengl_to_image(
                        width, height, self.mapped_main_cnt)
                    # dst_directory = Helper.get_or_create_image_directory_from_image_name(
                    #     directory, result_image_path)
                    Helper.save_contours_csv(temp_directory,
                                             mapped_back_to_image,
                                             result_image_path)
            except IndexError as e:
                tb = traceback.format_exc()
                print(tb)
                # contour = Helper.load_contours_from_csv(
                #     directory, result_image_path)
                mapped_back_to_image = self.map_from_opengl_to_image(
                    width, height, self.mapped_main_cnt)
                # dst_directory = Helper.get_or_create_image_directory_from_image_name(
                #     directory, result_image_path)
                Helper.save_contours_csv(temp_directory,
                                         mapped_back_to_image,
                                         result_image_path)

            # remove the resampled file if exists because the landmarks are changed!
            # result_image_name, _ = Helper.separate_file_name_and_extension(result_image_path)
            # search_term = result_image_name + '_resampled_result*'
            # folder_path = "/".join(result_image_path.split("/")[:-1])
            # Helper.remove_from_temp_directory(folder_path, filename=search_term)

            Helper.update_landmarks(self.mapped_main_cnt, temp_directory, result_image_path)

        if self.editing_mode == 2:
            data = Helper.read_metadata_from_csv(temp_directory, result_image_path)
            # landmarks = Helper.change_number_of_contour_coordinates(landmarks)
            landmarks_mapped_back_to_image = self.map_from_opengl_to_image(
                width, height, self.mapped_landmarks)
            try:
                data['landmarks'] = landmarks_mapped_back_to_image[0]
            except:
                pass

            Helper.save_metadata_to_csv(temp_directory, result_image_path, data)
