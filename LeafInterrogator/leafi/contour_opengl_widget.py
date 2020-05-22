import logging
import math
import traceback

import OpenGL.GL.shaders
import OpenGL.GLU as GLU
import freetype
import numpy as np
import pyrr
from OpenGL.GL import *
from PIL import Image
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import (QPixmap, QGuiApplication)
from PyQt5.QtWidgets import QDialog, QOpenGLWidget

#from .extension_scaleBarInputDialog_gui import ExtensionScaleBarInputDialog
from .helper import Helper

MINIMAL_VERSION = True

class ContourOpenGLWidget(QOpenGLWidget):
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
        super(ContourOpenGLWidget, self).__init__(parent)

        self.versionprofile = versionprofile

        self.logger = logging.getLogger(__name__)

        self.flfwt_object = None

        self.VAOs_list = []
        self.lines_vao_ref = []
        self.shader_program = None

        self.current_pos = QPoint()
        self.last_pos = QPoint()
        self.positions_list = []
        self.positions_list_scrolling = []
        self.scale_bar_text_data = None
        self.scale_bar_value = None
        self.scale_prefix = None

        # cut points and mapped cut points
        # of the current image (temporary)
        self.combined_cut_points = []
        self.mapped_cut_points = []

        self.mapped_contour_points = []
        # keep selected points for delete or change
        self.mapped_ready_to_change_cut_points = []
        self.selected_cut_points = []

        self.add_points_list = []
        # {'leaflets':[], 'terminal':[]}
        self.cut_points_dict = {}
        self.mapped_cut_points_dict = {}
        # image_name -> mapped_cut_points
        self.image_name_cut_points_mapping_dict = {}
        self.mapped_image_name_cut_points_mapping_dict = {}

        self.image_name_main_rachis_dict = {}

        self.main_rachis = None
        self.edit_leaflets_cut_points = False
        self.line_points = []
        self.mouse_x = None
        self.mouse_y = None

        self.image_path = None
        self.image = None
        self.current_texture_reference = None
        self.view_port = None
        self.matrices_flag = False

        self.model_view_matrix = None
        # self.projection_matrix = None
        self.model_view_projection_matrix = None
        self.straight_line_model_view_projection_matrix = None

        # model matrix is also identity
        self.model_matrix = pyrr.Matrix44.identity()
        # initialize view matrix
        self.view_matrix = pyrr.Matrix44.identity()
        # projection matrix is identity in our 2D case
        self.projection_matrix = pyrr.Matrix44.identity()

        self.translate_matrix = pyrr.Matrix44.identity()
        self.scale_matrix = pyrr.Matrix44.identity()

        self.counter = 0

        self.hand_cursor = False

        self.rotation_position = 0

        self.cut_points_color = [1.0, 0.0, 0.0]
        self.cut_points_size = 10.0
        self.selected_cut_points_color = [0.0, 0.0, 1.0]

        self.ready_to_change_cut_points_color = [0.0, 1.0, 0.0]

        self.add_button_state_cut_points = False
        self.select_button_state_cut_points = False

    def initializeGL(self):
        #print(self.width(), self.height())
        # In order to find the viewport we draw a white rectangle
        # in whole screen
        # self.create_bounding_rect()
        self.create_mvp_with_identity()
        # Ignore mouse and keyboard events
        self.setEnabled(False)
        # self.create_rectangle_with_texture(
        #     './data/ExampleLeafShapes/A alpina1.jpg')

    def clear_screen(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.VAOs_list = []
        self.image = None
        self.image_path = None
        self.update()

    def paintGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # self.counter +=1
        # print("counter= ",self.counter)
        try:
            for data in self.VAOs_list:
                object_id = data['object_id']
                glUseProgram(data['shader'])
                VAO = data['VAO']
                object_type = data['type']
                mvp_matrix = data['mvp_matrix']
                point_size = data['point_size']
                # check if the Vertex Array Object exist
                if glIsVertexArray(VAO) == 0:
                    self.update_opengl_widget()
                    break

                glBindVertexArray(VAO)
                # draw triangle
                if object_type == GL_TRIANGLES and (object_id ==
                                                        'main_texture' or
                                                            object_id is None):
                    try:
                        if self.image:
                            img_data = self.image.convert("RGBA").tobytes()

                            glBindTexture(GL_TEXTURE_2D, data['texture'])
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                         self.image.width,
                                         self.image.height, 0,
                                         GL_RGBA, GL_UNSIGNED_BYTE,
                                         img_data)
                    except:
                        pass

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
                    # glDrawArrays(GL_LINE_STRIP, 2, GL_UNSIGNED_INT, None)
                    glDrawArrays(GL_LINE_STRIP, 0, data['number_of_points'])
                elif object_type == GL_POINTS:
                    glPointSize(point_size)

                    # glEnable(GL_BLEND)
                    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

                    glUseProgram(data['shader'])
                    transform_loc = glGetUniformLocation(data['shader'],
                                                         "mvp_matrix")
                    glUniformMatrix4fv(transform_loc, 1, GL_FALSE,
                                       mvp_matrix)
                    glDrawArrays(GL_POINTS, 0, data['number_of_points'])

                elif object_id == 'scale_bar_text':
                    img_data = self.scale_bar_text_data.convert(
                        "RGBA").tobytes()

                    glBindTexture(GL_TEXTURE_2D, data['texture'])
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                                 self.scale_bar_text_data.width,
                                 self.scale_bar_text_data.height, 0,
                                 GL_RGBA, GL_UNSIGNED_BYTE,
                                 img_data)
                    transform_loc = glGetUniformLocation(data['shader'],
                                                         "mvp_matrix")
                    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, mvp_matrix)

                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
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
        self.VAOs_list = []
        # self.create_bounding_rect()
        if self.image_path:
            self.create_rectangle_with_texture(self.image_path)

        if len(self.line_points) == 2:
            self.create_line_between_points(self.line_points[0], self.line_points[1])

        if len(self.positions_list_scrolling) == 2:
            self.scroll_image()
        if len(self.positions_list) == 2:
            self.draw_line_with_width(0.02, self.positions_list[0],
                                      self.positions_list[1])
            if self.scale_bar_value:
                self.write_text_on_texture(self.positions_list[0],
                                           self.positions_list[1], 0.02)

        if len(self.mapped_cut_points) > 0:
            self.create_cut_points(
                self.mapped_cut_points, self.cut_points_color,
                self.cut_points_size)

        if len(self.mapped_ready_to_change_cut_points) > 0:
            self.create_cut_points(
                self.mapped_ready_to_change_cut_points,
                self.selected_cut_points_color,
                self.cut_points_size)

    def create_object(self, object_matrix=None, indices=None,
                      gldraw_element_type=None, image_path=None,
                      use_texture=False, object_id=None, point_size=1.0):
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

            self.image_path = image_path
            if self.image_path:
                try:
                    self.image = Image.open(self.image_path)
                    img_data = self.image.convert("RGBA").tobytes()
                except:
                    pass

                try:
                    pixmap = QPixmap(self.image_path)
                    pixmap.save("{}/temp.tif".format(self.temp_directory))

                    self.image = Image.open("{}/temp.tif".format(self.temp_directory))
                    img_data = self.image.convert("RGBA").tobytes()
                except Exception as e:
                    print("can not open the image!", e)

            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image.width,
                         self.image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         img_data)

        return {'VAO': VAO, 'type': gldraw_element_type, 'shader':
            self.shader_program, 'texture': texture, 'object_id': object_id,
                'point_size': point_size
                }

    def prepare_contour_points(self, contour_points, texture_width, texture_height):
        """

        :param contour_points: contour points from OpenCV findContours function.
        :param texture_width: Width of the Texture (image)
        :param texture_height: Height of the Texture (image)
        :return: -
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[3] = self.height()
        view_port[2] = self.width()

        texture_width = int(texture_width)
        texture_height = int(texture_height)

        self.mapped_contour_points = Helper.map_from_image_to_opengl(
            contour_points, view_port, texture_width, texture_height)

    def map_cut_points_to_opengl(self):
        """

        :param cut_points:
        :param texture_width: Width of the Texture (image)
        :param texture_height: Height of the Texture (image)
        :return: -
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        for image_name, cut_points_dict in self.image_name_cut_points_mapping_dict.items():
            mapped_cut_points = []
            for k, v in cut_points_dict.items():
                reshaped_cutpoints = [np.array(v).reshape(-1, 1, 2)]
                self.mapped_cut_points_dict[k] = Helper.map_from_image_to_opengl(
                    reshaped_cutpoints, view_port, self.texture_width, self.texture_height)
                mapped_cut_points.extend(np.vstack(self.mapped_cut_points_dict[k]).squeeze())

            self.mapped_cut_points = [np.array(mapped_cut_points).reshape(-1, 1, 3)]
            self.mapped_image_name_cut_points_mapping_dict[image_name] = self.mapped_cut_points

    def prepare_cut_points(self, temp_directory, image_name):
        """
        prepare cut points to view
        :param image_name:
        :return:
        """
        contour_points = Helper.load_contours_from_csv(temp_directory, self.get_image_name())
        self.prepare_contour_points(contour_points, self.texture_width, self.texture_height)

        combined_cut_points = self.image_name_cut_points_mapping_dict[image_name]['leaflets'].copy()
        combined_cut_points.extend(self.image_name_cut_points_mapping_dict[image_name]['terminal'])
        self.combined_cut_points = combined_cut_points
        self.mapped_ready_to_change_cut_points = []

        self.cut_points_dict = self.image_name_cut_points_mapping_dict[image_name]
        self.mapped_cut_points = self.mapped_image_name_cut_points_mapping_dict[image_name]

        view_port = glGetIntegerv(GL_VIEWPORT)
        for img_name, cut_points_dict in self.image_name_cut_points_mapping_dict.items():
            if img_name == image_name:
                for k, v in cut_points_dict.items():
                    reshaped_cutpoints = [np.array(v).reshape(-1, 1, 2)]
                    self.mapped_cut_points_dict[k] = Helper.map_from_image_to_opengl(
                        reshaped_cutpoints, view_port, self.texture_width, self.texture_height)
            self.current_image_name = image_name

        self.update_opengl_widget()

    def prepare_ready_to_change_cut_points(self, cut_points):
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[3] = self.height()
        view_port[2] = self.width()

        # self.mapped_ready_to_change_cut_points = Helper.map_from_image_to_opengl(
        #     cut_points, view_port, self.texture_width, self.texture_height)
        # if cut_points:
        self.mapped_ready_to_change_cut_points = cut_points

        self.update_opengl_widget()

    def create_cut_points(self, cut_points, color, points_size):
        """
        draw cut points on image
        :return:
        """

        model_view_projection_matrix = self.create_model_view_matrix()
        for point in cut_points[0]:
            object_matrix = self.create_object_matrix(point, color)

            object_matrix = np.array(object_matrix, dtype=np.float32)

            data_dict = self.create_object(object_matrix, None, GL_POINTS,
                                           point_size=points_size)

            model_view_matrix = model_view_projection_matrix

            data_dict['mvp_matrix'] = model_view_matrix
            data_dict['number_of_points'] = int(len(object_matrix) / 8)

            self.VAOs_list.append(data_dict)

    def create_object_matrix(self, point, color_rgb):
        """
        create object matrix from the set of points
        :param point: [[x1,y1,z1]]
        :param color_rgb: [r, g, b]
        :return:
        """
        object_matrix = []

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

    def create_rectangle_with_texture(self, texture_path):
        if not texture_path:
            return 1
        self.setEnabled(True)
        #                positions        colors         texture coords
        object_matrix = [-1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                         # Top-left
                         1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                         # Top-right
                         1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                         # Bottom-right
                         -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                         # Bottom-left
                         ]

        object_matrix = np.array(object_matrix, dtype=np.float32)

        indices = [0, 1, 2,
                   2, 3, 0]
        indices = np.array(indices, dtype=np.int32)
        data_dict = self.create_object(object_matrix, indices, GL_TRIANGLES,
                                       texture_path, True,
                                       object_id='main_texture')
        self.texture_width = self.image.width
        self.texture_height = self.image.height

        model_view_projection_matrix = self.create_model_view_matrix()
        self.matrices_flag = True

        data_dict['mvp_matrix'] = model_view_projection_matrix
        self.VAOs_list[:] = []
        self.VAOs_list.append(data_dict)
        self.update()

    def write_text_on_texture(self, last_pos, current_pos, line_width):
        """
        Writing text on a texture
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
        # model_view_projection_matrix = pyrr.matrix44.multiply(
        #     self.model_view_matrix, self.projection_matrix)
        model_view_projection_matrix = self.create_model_view_matrix()

        if last_pos[0] > current_pos[0]:
            temp = last_pos
            last_pos = current_pos
            current_pos = temp

        length = (current_pos[0] - last_pos[0]) * 0.15
        line_y = last_pos[1]
        x0 = (last_pos[0] + length, line_y + 0.02)  # bottom left
        x1 = (current_pos[0] - length, line_y + 0.02)  # bottom right
        x2 = (current_pos[0] - length, line_y + line_width + 0.15)  # top right
        x3 = (last_pos[0] + length, line_y + line_width + 0.15)  # top left

        # print(x0)
        # # Horizontal
        # if abs(last_pos[0] - current_pos[0]) < abs(last_pos[1] -
        #                                                    current_pos[1]):
        #     rotate90 = np.array([[0, -1], [1, 0]])
        #     # print(np.array(x0).reshape(2, 1))
        #     x0 = np.dot(np.array(x0).reshape(1, 2), rotate90)[0]
        #     x1 = np.dot(np.array(x1).reshape(1, 2), rotate90)[0]
        #     x2 = np.dot(np.array(x2).reshape(1, 2), rotate90)[0]
        #     x3 = np.dot(np.array(x3).reshape(1, 2), rotate90)[0]
        # print(x0)
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

        face = freetype.Face('./fonts/Vera.ttf')
        text = self.scale_bar_value + self.scale_prefix
        face.set_char_size(128 * 128)
        slot = face.glyph

        # First pass to compute bbox
        width, height, baseline = 0, 0, 0
        previous = 0
        for i, c in enumerate(text):
            face.load_char(c)
            bitmap = slot.bitmap
            height = max(height,
                         bitmap.rows + max(0,
                                           -(slot.bitmap_top - bitmap.rows)))
            baseline = max(baseline, max(0, -(slot.bitmap_top - bitmap.rows)))
            kerning = face.get_kerning(previous, c)
            width += (slot.advance.x >> 6) + (kerning.x >> 6)
            previous = c

        z = np.zeros((height, width), dtype=np.ubyte)

        # Second pass for actual rendering
        x, y = 0, 0
        previous = 0
        for c in text:
            face.load_char(c)
            bitmap = slot.bitmap
            top = slot.bitmap_top
            left = slot.bitmap_left
            w, h = bitmap.width, bitmap.rows
            y = height - baseline - top
            kerning = face.get_kerning(previous, c)
            x += (kerning.x >> 6)
            z[y:y + h, x:x + w] += np.array(bitmap.buffer,
                                            dtype='ubyte').reshape(h, w)
            x += (slot.advance.x >> 6)
            previous = c

        z = Helper.change_scale_indicator_bg_color(z)

        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        self.shader_program = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(self.VERTEX_SHADER,
                                            GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(self.FRAGMENT_SHADER_TEX,
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

        img = Image.fromarray(z)
        imd = img.convert("RGBA").tobytes()

        glBindTexture(GL_TEXTURE_2D, texture)
        # glBitmap(width, width, 0, 0, 0, 0, text_bitmap)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width,
                     img.height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     imd)
        self.scale_bar_text_data = img

        data_dict = {'VAO': VAO, 'type': GL_TRIANGLES,
                     'shader': self.shader_program, 'texture': texture,
                     'mvp_matrix': model_view_projection_matrix,
                     'object_id': 'scale_bar_text',
                     'point_size': 1.0}

        self.VAOs_list.append(data_dict)

    def get_pair_of_point(self, input_point, cut_points_list):
        """
        find pair of the input point
        :param input_point: list or ndarray
                input point should be in openGL coordinate (the values are between 0 and 1)
        :param cut_points_list: list
        :return: ndarray
            pair of points
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[3] = self.height()
        view_port[2] = self.width()

        mapped_input_point = self.map_from_opengl_to_image(input_point[0],
                                                           self.texture_width,
                                                           self.texture_height)
        mapped_input_point = mapped_input_point[:2]
        for pair_point in cut_points_list:
            for p in pair_point:
                if np.all(np.abs(np.array(mapped_input_point - p)) < 0.01):
                    point = Helper.map_from_image_to_opengl(
                        [np.array(pair_point).reshape(-1, 1, 2)],
                        view_port, self.texture_width,
                        self.texture_height)

                    return point
        return np.array([])

    def mousePressEvent(self, event):
        self.setFocus()
        self.mouse_x = event.pos().x()
        self.mouse_y = event.pos().y()
        modifiers = QGuiApplication.keyboardModifiers()
        if not self.is_point_on_image(self.mouse_x, self.mouse_y):
            return

        try:
            x_new, y_new, z_new = self.map_qt_to_opengl_coordinates(
                self.mouse_x, self.mouse_y)
        except:
            return

        if modifiers == Qt.AltModifier:
            self.setCursor(Qt.ClosedHandCursor)
            if len(self.positions_list_scrolling) == 0:
                self.positions_list_scrolling.append((x_new, y_new))

        if MINIMAL_VERSION == False and self.edit_leaflets_cut_points:
            if (QGuiApplication.keyboardModifiers() == Qt.ShiftModifier) or \
                    self.select_button_state_cut_points:
                nearest_point = Helper.nearest_point(
                    self.mapped_cut_points, (x_new, y_new))
                if nearest_point is None:
                    return
                if not nearest_point.any():
                    return

                for k, v in self.cut_points_dict.items():
                    self.selected_cut_points = self.get_pair_of_point(
                        nearest_point, v)

                    if self.selected_cut_points:
                        break
                self.selected_cut_points = [np.array(self.selected_cut_points).reshape(-1, 1, 3)]

                self.prepare_ready_to_change_cut_points(self.selected_cut_points)

            if (modifiers == (Qt.ControlModifier | Qt.ShiftModifier)) or \
                    self.add_button_state_cut_points:
                try:
                    self.mapped_contour_points = [np.array(self.mapped_contour_points).reshape((-1, 1, 3))]
                    new_point = Helper.nearest_point(
                        self.mapped_contour_points, (x_new, y_new))
                except:
                    return

                if new_point is None:
                    return
                self.add_points_list.append(new_point)

                try:
                    temp = self.mapped_cut_points[0].tolist()
                except IndexError as e:
                    print('Error:', e)
                    temp = []
                for ap in self.add_points_list:
                    temp.append(ap.tolist())

                self.mapped_cut_points = [np.array(temp).reshape(-1, 1, 3)]
                if len(self.add_points_list) == 1:
                    self.setMouseTracking(True)

                if len(self.add_points_list) == 2:
                    view_port = glGetIntegerv(GL_VIEWPORT)
                    view_port[3] = self.height()
                    view_port[2] = self.width()
                    mapped_back_to_image_points_list = []

                    for ap in self.add_points_list:
                        mapped_back_to_image_point = self.map_from_opengl_to_image(
                            ap[0], self.texture_width, self.texture_height)

                        mapped_back_to_image_points_list.append(
                            np.array(list(mapped_back_to_image_point[:2]),
                                     dtype=np.float32))

                    mapped_back_to_image_points_tuple = tuple(mapped_back_to_image_points_list)
                    self.combined_cut_points.append(mapped_back_to_image_points_tuple)

                    main_rachis_coords = np.flip(self.image_name_main_rachis_dict[
                                                     self.current_image_name], axis=1)

                    are_terminal_bool = Helper.are_terminal_cut_points(
                        mapped_back_to_image_points_tuple, main_rachis_coords,
                        self.texture_width, self.texture_height)
                    # are_terminal_bool = self.flfwt_object.dls.are_terminal_cut_points(
                    #     mapped_back_to_image_points_tuple, main_rachis_coords)

                    # now remove cut points on a same leaflet
                    temp = []
                    if are_terminal_bool:
                        # for ap in self.mapped_cut_points[0].tolist():
                        #     if ap not in self.mapped_cut_points_dict['terminal'][0].tolist():
                        #         temp.append(ap)
                        # self.mapped_cut_points = [np.array(temp).reshape(-1, 1, 3)]

                        self.cut_points_dict['terminal'] = [mapped_back_to_image_points_tuple]
                    else:
                        # TODO: For leaflets it is Not clear which one should be removed ??!!
                        # for ap in self.mapped_cut_points[0].tolist():
                        #     if ap not in self.mapped_cut_points_dict['leaflets'][0].tolist():
                        #         temp.append(ap)
                        # self.mapped_cut_points = [np.array(temp).reshape(-1, 1, 3)]

                        self.cut_points_dict['leaflets'].append(mapped_back_to_image_points_tuple)

                    self.line_points = [x[0] for x in self.add_points_list]
                    self.add_points_list = []
                    self.setMouseTracking(False)

                    self.update_opengl_widget()
                elif len(self.add_points_list) == 1:
                    pass
                self.update_opengl_widget()

        elif MINIMAL_VERSION == False and modifiers == Qt.ShiftModifier:

            self.positions_list.append((x_new, y_new))

            if len(self.positions_list) == 2:
                # open a popup in order to get the scale bar line value
                dialog = QDialog()
                dialog.ui = ExtensionScaleBarInputDialog()
                dialog.ui.setupUi(dialog)
                dialog.exec_()

                if not dialog.ui.reject_button_clicked:
                    # delete the attributes of popup dialog
                    # dialog.setAttribute(Qt.WA_DeleteOnClose)

                    self.draw_line_with_width(0.02, self.positions_list[0],
                                              self.positions_list[1])
                    self.update_opengl_widget()

                    # get the input value
                    self.scale_bar_value = dialog.ui.scale_input_value.text()
                    self.scale_prefix = \
                        dialog.ui.metric_prefixes_combobox.currentText()

                    self.write_text_on_texture(self.positions_list[0],
                                               self.positions_list[1], 0.02)
                    self.positions_list = []

                else:
                    self.positions_list = []

            elif len(self.positions_list) == 1:
                self.scale_bar_value = None
                self.scale_prefix = None

    def mouseMoveEvent(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()
        if not self.is_point_on_image(mouse_x, mouse_y):
            return

        x_new, y_new, z_new = self.map_qt_to_opengl_coordinates(mouse_x,
                                                                mouse_y
                                                                )
        if MINIMAL_VERSION == False and self.edit_leaflets_cut_points:
            # === draw line from point to mouse cursor
            if len(self.add_points_list) == 1 and (
                        (modifiers == (Qt.ControlModifier | Qt.ShiftModifier)) or
                        self.add_button_state_cut_points):
                # print(self.add_points_list[0][0])
                self.line_points = [self.add_points_list[0][0], [x_new, y_new, z_new]]

                self.update_opengl_widget()
                # ======================

        if QGuiApplication.keyboardModifiers() == Qt.AltModifier:
            self.positions_list_scrolling.append((x_new, y_new))
            if len(self.positions_list_scrolling) >= 2:
                self.scroll_image()

        else:
            if not self.is_point_on_image(event.pos().x(), event.pos().y()) \
                    or not self.positions_list:
                return
            x_new, y_new, z_new = self.map_qt_to_opengl_coordinates(
                event.pos().x(), event.pos().y())

            if self.positions_list[0][0] != x_new or \
                            self.positions_list[0][1] != y_new:

                # print(self.positions_list)
                if (self.euclidean_distance(self.positions_list[0],
                                            (x_new, y_new)) >= 0.01):
                    self.positions_list.append((x_new, y_new))
                    self.draw_line_with_width(0.01, self.positions_list[0],
                                              self.positions_list[1])

                    self.update_opengl_widget()

                    self.positions_list.pop()

    def mouseReleaseEvent(self, event):
        self.positions_list_scrolling = []
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, qwheelevent):
        if qwheelevent.angleDelta().y() > 0:
            self.zoom_in_opengl_widget(1.2)
        if qwheelevent.angleDelta().y() < 0:
            self.zoom_in_opengl_widget(0.75)

    def keyReleaseEvent(self, QKeyEvent):
        modifiers = QGuiApplication.keyboardModifiers()

        if MINIMAL_VERSION == False and self.edit_leaflets_cut_points:
            if len(self.add_points_list) == 1 and (
                        modifiers == (Qt.ControlModifier | Qt.ShiftModifier)):

                temp = self.mapped_cut_points[0].tolist()
                for ap in self.add_points_list:
                    temp.remove(ap.tolist())
                self.mapped_cut_points = [np.array(temp).reshape(-1, 1, 3)]

                self.line_points = []
                self.add_points_list = []

                self.update_opengl_widget()

    def keyPressEvent(self, QKeyEvent):
        if MINIMAL_VERSION == False and self.edit_leaflets_cut_points:
            # Delete the point with Backspace key on the keyboard
            if QKeyEvent.key() == Qt.Key_Backspace:
                self.remove_selected_cut_points()
                self.update_opengl_widget()

        if not self.hand_cursor and QKeyEvent.key() == Qt.Key_Alt:
            self.setCursor(Qt.OpenHandCursor)
            self.hand_cursor = True
        else:
            self.setCursor(Qt.ArrowCursor)
            self.hand_cursor = False

    def remove_from_cut_points(self, points_list, point):
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[3] = self.height()
        view_port[2] = self.width()
        mapped_back_points_list = []
        mapped_back_point = self.map_from_opengl_to_image(point,
                                                          self.texture_width,
                                                          self.texture_height)
        mapped_back_points_list.append(np.array(list(mapped_back_point[:2]), dtype=np.float32))
        temp = []
        for tp in points_list:
            count = 0
            for p in tp:
                count += 1
                if np.all(abs(mapped_back_points_list[0] - p)) < 0.0001:
                    break
                elif count == 2:
                    temp.append(tp)
        points_list = temp

        return points_list

    def remove_selected_cut_points(self):
        temp = []
        for k, v in self.cut_points_dict.items():
            try:
                selected_list = self.selected_cut_points[0].tolist()
            except IndexError:
                self.logger.error("Index error!")
                return
            cut_points_list = self.mapped_cut_points_dict[k][0].tolist()

            for point in selected_list:
                try:
                    cut_points_list.remove(point[0])
                    self.combined_cut_points = self.remove_from_cut_points(self.combined_cut_points, point[0])
                except Exception:
                    self.logger.error('No cut points')
                    pass

            for point in cut_points_list:
                temp.append(point)
            self.mapped_cut_points_dict[k] = [np.array(cut_points_list)]

        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[3] = self.height()
        view_port[2] = self.width()

        for k, v in self.mapped_cut_points_dict.items():
            mapped_back_to_image_points_list = []
            counter = 0
            l = []
            for point in v[0]:
                mapped_back_to_image_point = self.map_from_opengl_to_image(point,
                                                                           self.texture_width,
                                                                           self.texture_height)
                mapped_back_to_image_points_list.append(
                    np.array(list(mapped_back_to_image_point[:2]),
                             dtype=np.float32))
                counter += 1
                if counter % 2 == 0:
                    mapped_back_to_image_points_tuple = tuple(mapped_back_to_image_points_list)
                    l.append(mapped_back_to_image_points_tuple)
                    mapped_back_to_image_points_list = []
            self.cut_points_dict[k] = l

        self.mapped_cut_points = [np.array(temp).reshape(-1, 1, 3)]
        self.mapped_ready_to_change_cut_points = []
        self.selected_cut_points = []

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

    def create_line_between_points(self, last_pos, current_pos):
        """
        create line object. it will draw line between two points.
        :param last_pos:
        :param current_pos:
        :return:
        """

        points = [[last_pos], [current_pos]]

        color_rgb = [0.0, 1.0, 0.0]
        object_matrix = self.create_line_object_matrix(points,
                                                       color_rgb)

        object_matrix = np.array(object_matrix, dtype=np.float32)

        data_dict = self.create_object(object_matrix, None, GL_LINE_STRIP)

        mv_matrix = self.create_model_view_matrix()

        data_dict['mvp_matrix'] = mv_matrix
        data_dict['number_of_points'] = int(len(object_matrix) / 8)

        self.VAOs_list.append(data_dict)

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

        return object_matrix

    def angle(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        inner_product = x1 * x2 + y1 * y2
        len1 = math.hypot(x1, y1)
        len2 = math.hypot(x2, y2)
        return math.acos(inner_product / (len1 * len2))

    def draw_line_with_width(self, line_width, last_pos, current_pos):
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
        # model_view_projection_matrix = pyrr.matrix44.multiply(
        #     self.model_view_matrix, self.projection_matrix)
        model_view_projection_matrix = self.create_model_view_matrix()

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
            data_dict['mvp_matrix'] = model_view_projection_matrix

            self.VAOs_list.append(data_dict)

        # Vertical line
        elif abs(last_pos[0] - current_pos[0]) < abs(last_pos[1] -
                                                             current_pos[1]):
            if ratio < 1:
                line_width *= ratio
            else:
                line_width /= ratio
            line_x = last_pos[0]
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

            data_dict['mvp_matrix'] = model_view_projection_matrix

            self.VAOs_list.append(data_dict)

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

    def create_model_view_matrix(self):
        """
        create model view matrix for the current image (contour)
        :return: 4x4 mvp_matrix
        """
        view_port = glGetIntegerv(GL_VIEWPORT)
        view_port[2] = self.width()
        view_port[3] = self.height()

        texture_width = self.texture_width
        texture_height = self.texture_height

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
        self.projection_matrix = pyrr.Matrix44.identity()

        model_view_matrix = pyrr.matrix44.multiply(self.scale_matrix,
                                                   model_view_matrix)

        model_view_matrix = pyrr.matrix44.multiply(self.translate_matrix,
                                                   model_view_matrix)

        return model_view_matrix

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
            self.logger.error('No contour founded', exc_info=True)
            return all_cnt_coordinates
        coordinate = []
        for point in cnt:
            x = point[0][0]
            y = point[0][1]
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
        widget_x, widget_y, widget_z = GLU.gluProject(x, y, z,
                                                      self.model_view_matrix,
                                                      self.projection_matrix,
                                                      view_port)

        return widget_x, widget_y, widget_z

    def map_from_opengl_to_image(self, point_3d, width, height):

        x, y, z = point_3d
        view_port = glGetIntegerv(GL_VIEWPORT)
        object_matrix = []
        # contour_np = [[width ,height]]
        view_port[2] = width
        view_port[3] = height
        self.model_view_matrix = pyrr.Matrix44.identity()

        x_new, y_new, z_new = GLU.gluProject(x, y * -1, z,
                                             self.model_view_matrix,
                                             self.projection_matrix,
                                             view_port)

        mapped_point = (x_new, y_new, z_new)

        return mapped_point

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
            # if data['type'] == GL_POINTS or data['type'] == GL_LINE_STRIP:
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
            # if data['type'] == GL_POINTS or data['type'] == GL_LINE_STRIP:
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
            # if data['type'] == GL_POINTS or data['type'] == GL_LINE_STRIP:
            data['mvp_matrix'] = pyrr.matrix44.multiply(translate_m,
                                                        data['mvp_matrix'])
            self.update()
            self.positions_list_scrolling = []

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return dist

    def rotate(self, rotate_direction="anticlockwise"):
        if rotate_direction == "anticlockwise":
            rotation_matrix_z = pyrr.Matrix44.identity()
            rotation_matrix_z[0][0] = 0
            rotation_matrix_z[0][1] = -1
            rotation_matrix_z[1][0] = 1
            rotation_matrix_z[1][1] = 0
        elif rotate_direction == "clockwise":
            rotation_matrix_z = pyrr.Matrix44.identity()
            rotation_matrix_z[0][0] = 0
            rotation_matrix_z[0][1] = 1
            rotation_matrix_z[1][0] = -1
            rotation_matrix_z[1][1] = 0

        for data in self.VAOs_list:
            if data['type'] == GL_TRIANGLES:
                # print(data['mvp_matrix'])
                # if self.image.width > self.image.height:
                if data['mvp_matrix'][0][0] > 0 or data['mvp_matrix'][1][1] \
                        > 0:
                    image_ratio = self.image.width / self.image.height
                    screen_ratio = self.width() / self.height()
                    view_matrix = pyrr.Matrix44().identity()
                    if image_ratio > screen_ratio:
                        view_matrix[0][0] = -1 / screen_ratio * 1 / image_ratio
                        view_matrix[1][1] = -1

                    else:
                        view_matrix[0][0] = -1
                        view_matrix[1][1] = -1 * screen_ratio / (1 / image_ratio)
                    view_matrix[2][2] = -1

                elif data['mvp_matrix'][0][0] == 0 and (data['mvp_matrix'][
                                                            0][1] > 0 or data['mvp_matrix'][1][0] < 0):
                    image_ratio = self.image.width / self.image.height
                    screen_ratio = self.width() / self.height()
                    view_matrix = pyrr.Matrix44().identity()
                    if image_ratio > screen_ratio:
                        view_matrix[0][0] = 0
                        view_matrix[0][1] = -1 * screen_ratio / image_ratio
                        view_matrix[1][0] = 1
                        view_matrix[1][1] = 0

                    else:
                        view_matrix[0][0] = 0
                        view_matrix[0][1] = -1
                        view_matrix[1][0] = 1 / screen_ratio * image_ratio
                        view_matrix[1][1] = 0
                    view_matrix[2][2] = -1

                elif (data['mvp_matrix'][0][0] == 0 and data['mvp_matrix'][
                    0][1] < 0) or (data['mvp_matrix'][1][0] == 0 and data[
                    'mvp_matrix'][1][1] < 0 and data['mvp_matrix'][0][0] == 0):
                    image_ratio = self.image.width / self.image.height
                    screen_ratio = self.width() / self.height()
                    view_matrix = pyrr.Matrix44().identity()
                    if image_ratio > screen_ratio:
                        view_matrix[0][0] = 0
                        view_matrix[0][1] = 1 * screen_ratio / image_ratio
                        view_matrix[1][0] = - 1
                        view_matrix[1][1] = 0

                    else:
                        view_matrix[0][0] = 0
                        view_matrix[0][1] = 1
                        view_matrix[1][0] = -1 / screen_ratio * image_ratio
                        view_matrix[1][1] = 0
                    view_matrix[2][2] = -1
                elif data['mvp_matrix'][0][0] < 0 and data['mvp_matrix'][1][
                    1] < 0:
                    image_ratio = self.image.width / self.image.height
                    screen_ratio = self.width() / self.height()
                    view_matrix = pyrr.Matrix44().identity()
                    if image_ratio > screen_ratio:
                        view_matrix[0][0] = 1 / screen_ratio * 1 / image_ratio
                        view_matrix[1][1] = 1
                    else:
                        view_matrix[0][0] = 1
                        view_matrix[1][1] = 1 * screen_ratio / (
                            1 / image_ratio)
                    view_matrix[2][2] = -1
                # view_matrix = data['mvp_matrix']
                elif data['mvp_matrix'][0][0] == 0 and data['mvp_matrix'][
                    1][0] > 0:
                    image_ratio = self.image.width / self.image.height
                    screen_ratio = self.width() / self.height()
                    view_matrix = pyrr.Matrix44().identity()
                    if image_ratio > screen_ratio:
                        pass
                    else:
                        view_matrix[0][0] = 0
                        view_matrix[0][1] = 1
                        view_matrix[1][0] = - 1 / screen_ratio * image_ratio
                        view_matrix[1][1] = 0
                    view_matrix[2][2] = -1

                data['mvp_matrix'] = pyrr.matrix44.multiply(rotation_matrix_z,
                                                            view_matrix)

        self.update()

    def save_scene_as_image(self, path):
        glReadBuffer(GL_FRONT)
        # view_port = glGetIntegerv(GL_VIEWPORT)
        # print(view_port, self.pos(),self.geometry(), self.frameGeometry())

        x = self.geometry().x()
        y = self.geometry().y()

        width = self.geometry().width() * self.devicePixelRatio()
        height = self.geometry().height() * self.devicePixelRatio()

        data = glReadPixels(50, 92, width, height,
                            GL_RGB,
                            GL_UNSIGNED_BYTE)
        img = Image.frombytes(self.image.mode, (width, height),
                              data).transpose(Image.FLIP_TOP_BOTTOM)

        img.save(path)

    def get_image_name(self):
        img_name, _ = Helper.separate_file_name_and_extension(self.image_path, True)
        return img_name
