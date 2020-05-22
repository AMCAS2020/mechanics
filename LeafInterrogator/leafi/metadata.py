from .helper import Helper


class Metadata:
    def __init__(self, temp_directory, result_image_path, **kwargs):
        # self.__dict__.update(kwargs)
        self.temp_directory = temp_directory
        self.result_image_path = result_image_path
        try:
            self.metadata_dict = Helper.read_metadata_from_csv(temp_directory, result_image_path)
            self.update_metadata(kwargs)
        except:

            self.metadata_dict = dict.fromkeys(['result_image_path',
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

            self.metadata_dict['result_image_path'] = result_image_path

            height, width = Helper.get_image_height_width(result_image_path)
            self.metadata_dict['image_width'] = width
            self.metadata_dict['image_height'] = height

            result_img_name, _ = Helper.separate_file_name_and_extension(result_image_path,
                                                                         keep_extension=True)
            self.metadata_dict['result_image_name'] = result_img_name

            self.metadata_dict.update(kwargs)

    def get_class_name(self):
        try:
            return self.metadata_dict[self.metadata_dict['as_class_param']]
        except KeyError:
            return None

    def save_metadata(self):

        temp = dict((k, v) for k, v in self.metadata_dict.items() if k)
        self.metadata_dict = temp
        Helper.save_metadata_to_csv(self.temp_directory,
                                    self.result_image_path,
                                    self.metadata_dict)

    def update_metadata(self, meta_dict):
        """
        Update the old metadata dictionary.

        NOTE: if the value exist in the metadata, it will be replaced with the new value!

        :param meta_dict:
        :return:
        """
        temp = dict((k, v) for k, v in meta_dict.items() if k)
        self.metadata_dict.update(temp)

    def add_metadata(self, meta_dict):
        """
        This function add the input dictionary to old metadata

        NOTE: if the value exist in the metadata, it will NOT change!

        :param meta_dict:
        :return:
        """
        for k, v in meta_dict.items():
            if not k:
                continue
            try:
                if not self.metadata_dict[k]:
                    self.metadata_dict[k] = v
            except KeyError:
                self.metadata_dict[k] = v
