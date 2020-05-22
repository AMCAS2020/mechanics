import numpy as np
from PyQt5 import QtWidgets

from .extension_matplotlibMainWindow_gui import ExtensionMatplotlibMainWindow


class AbstractProcessMeta(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call Process(*args, **kwargs) """
        obj = type.__call__(cls, *args, **kwargs)
        obj.check_properties()
        obj.check_methods()
        return obj


class Process(metaclass=AbstractProcessMeta):
    """
        All implemented processes must inherit from this class.
        The child classes must have _need_resampling, _process_name and _parent_name
        attributes (properties). Except these three attributes all other public
        attributes will be appear in parameter table. (Public variables
        should not start with '_' or '__')

        The child classes also must override the run function, which return a
        dictionary of {process_name: result_value}.

    """

    def __init__(self):
        self._need_resampling = None
        self._process_name = None
        self._parent_name = None
        self._plot_type = None  # 'line_plot' or 'bar_plot' or None

    def check_properties(self):
        if self._process_name is None:
            raise NotImplementedError("Please assign a value to process_name "
                                      "property!")
        if self._need_resampling is None:
            raise NotImplementedError("Please assign a bool value to 'need_resampling' "
                                      "property!")
        if self._parent_name is None:
            raise NotImplementedError("Please assign a value to 'parent_name' "
                                      "property! If you want the "
                                      "process be the root of the tree, you can "
                                      "assign it to be None.")

    def check_methods(self):
        this_run = getattr(self, "run").__func__
        base_run = getattr(Process, "run")

        if base_run is this_run:
            raise NotImplementedError("you did not implement 'run' method in class '{}' !".format(
                self.__class__.__name__))

            # this_graph = getattr(self, "plot_results").__func__
            # base_graph = getattr(Process, "plot_results")
            #
            # if base_graph is this_graph:
            #     raise NotImplementedError("you did not implement 'plot_results' method!")

    @property
    def need_resampling(self):
        return self._need_resampling

    @property
    def process_name(self):
        return self._process_name

    @property
    def parent_name(self):
        return self._parent_name

    def run(self) -> dict:
        """ Abstract rotate that must be implemented in subclass"""
        pass
        # raise NotImplementedError("you did not implement run method!")

    def plot_results(self, reference, data_dict_list):
        """
        This is the default function for drawing process results in Matplotlib canvas.

        :param reference:
        :param data_dict_list:
        :return:
        """
        class_vs_result_dict = {}
        class_color = {}
        for d in data_dict_list:
            if d['class_value'] in class_vs_result_dict.keys():
                class_vs_result_dict[d['class_value']].append(d['data'])
            else:
                class_vs_result_dict[d['class_value']] = [d['data']]

            class_color[d['class_value']] = d['color_rgb']

        x_label = ''
        y_label = "{} value".format(self.process_name)
        for key in class_color.keys():
            try:
                if self._plot_type is 'bar_plot':
                    x_label = ''
                    std_error = np.std(np.array(class_vs_result_dict[key])) / np.sqrt(len(class_vs_result_dict[key]))
                                       
                    class_vs_result_dict[key] = {'bar_height': np.mean(np.array(class_vs_result_dict[key])),
                                                 'yerr': std_error}
                elif self._plot_type is 'line_plot':
                    x_label = 'Sample points'
                    class_vs_result_dict[key] = {'mean': np.mean(np.array(class_vs_result_dict[key])[:], axis=0)}
            except ValueError:
                print('Check the shape of the arrays for class {}!'.format(key))
                continue

        _options = {
            'data_options': class_vs_result_dict,
            'plot_options':
                {
                    "label": None,
                    "title": self.process_name,
                    "use_grid": False,
                    "x_label": x_label,
                    "y_label": y_label,
                    "class_color": class_color,
                    # "legend_loc": 'upper left',
                }
        }

        if self._plot_type is 'bar_plot':
            reference.ui.plot_canvas.mpl_bar_plot(**_options)

        elif self._plot_type is 'line_plot':
            reference.ui.plot_canvas.mpl_line_plot(**_options)

    def draw_graph_on_widget(self, reference, data_dict_list, save_path):
        """
        Note: Please Do not override this function!! In order to draw different types of
        plots, override the 'plot_results' method!
        :param reference:
        :param data_dict_list:
        :param save_path:
        :return:
        """
        try:
            if self._plot_type is None:
                return
        except AttributeError:
            return
        # =====================
        reference.window = QtWidgets.QMainWindow()
        reference.ui = ExtensionMatplotlibMainWindow(reference.window)
        # ===================

        self.plot_results(reference, data_dict_list)

        reference.window.setWindowTitle("{} plot".format(self.process_name))
        reference.ui.save_plot(save_path)
        reference.window.show()
