import matplotlib
import numpy as np
from PyQt5 import QtCore, QtWidgets

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(right=0.6)
        # self.axes = self.fig.add_axes([0.1, 0.1, 0.5, 0.8])
        # We want the axes cleared every time plot() is called
        self.axes.clear()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)

        FigureCanvas.updateGeometry(self)


class LocalizedMplPlots(MplCanvas):
    """A canvas to draw one-dimensional functions for shape representation"""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)

    def mpl_bar_plot(self, **kwargs):
        """

        :param data_dict: Dictionary of the form {'class name': value}
        :param kwargs:
        :return:
        """
        plot_options = kwargs.get('plot_options', {})
        data_dict = kwargs.get('data_options', {})
        if not data_dict:
            return

        self.x_lbl = plot_options.get('x_label', None)
        self.y_lbl = plot_options.get('y_label', None)
        class_color = plot_options.get('class_color', None)
        fig_title = plot_options.get('title', "Plot")
        self.show_legend = plot_options.get('show_legend', False)
        self.legend = plot_options.get('legend', None)
        self.legend_loc = plot_options.get('legend_loc', 'best')
        use_grid = plot_options.get('use_grid', False)
        bar_width = kwargs.get('bar_width', 0.35)  # the width of the bars
        fontdict = plot_options.get('fontdict', None)
        minor = plot_options.get('minor', False)
        rotation = plot_options.get('rotation', 45)
        yerr_capsize = plot_options.get('yerr_capsize', 5)
        elinewidth = plot_options.get('elinewidth', 3)
        ecolor = plot_options.get('ecolor', 'black')

        show_std_error_bar = plot_options.get('show_std_error_bar', True)
        self.x_axis_lbl = self.axes.set_xlabel(self.x_lbl)
        self.y_axis_lbl = self.axes.set_ylabel(self.y_lbl)
        self.axes.grid(use_grid)
        self.fig_title = self.fig.suptitle(fig_title)
        self.axes.margins(0.05)

        class_height_dict = {}
        height_std = []

        for k, v in data_dict.items():
            class_height_dict[k] = v['bar_height']
            height_std.append((v['bar_height'], v['yerr']))

        sorted_data_keys, sorted_height_values = zip(*sorted(
            class_height_dict.items(), key=lambda x: x[1], reverse=True))
        index = np.arange(len(sorted_data_keys))
        sorted_height_stderr = np.array(sorted(height_std, key=lambda x: x[0], reverse=True))
        if show_std_error_bar:
            bar_list = self.axes.bar(index, sorted_height_stderr[:, 0],
                                     width=bar_width, yerr=sorted_height_stderr[:, 1],
                                     error_kw=dict(elinewidth=elinewidth, ecolor=ecolor,
                                                   capsize=yerr_capsize, capthick=2)
                                     )
        else:
            bar_list = self.axes.bar(index, sorted_height_stderr[:, 0], width=bar_width)

        if class_color:
            for i, key in enumerate(sorted_data_keys):
                bar_list[i].set_color(tuple(class_color[key]))

        self.axes.set_xticks(index)
        self.axes.set_xticklabels(sorted_data_keys, fontdict=fontdict, minor=minor, rotation=rotation)

        if self.show_legend:
            handles, labels = self.axes.get_legend_handles_labels()
            self.legend = self.axes.legend(handles, labels, loc=self.legend_loc)

        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.90)

    def mpl_line_plot(self, data_dict, **kwargs):
        self.x_lbl = kwargs.get('x_label', None)
        self.y_lbl = kwargs.get('y_label', None)
        class_color = kwargs.get('class_color', 'r')
        fig_title = kwargs.get('title', "Plot")
        self.show_legend = kwargs.get('show_legend', True)
        self.legend_loc = kwargs.get('legend_loc', 'best')
        self.legend = kwargs.get('legend', None)
        use_grid = kwargs.get('use_grid', False)

        self.x_axis_lbl = self.axes.set_xlabel(self.x_lbl)
        self.y_axis_lbl = self.axes.set_ylabel(self.y_lbl)
        self.axes.grid(use_grid)
        # self.axes.hold(use_hold)
        # self.axes.set_ylim([-1, 3])

        self.fig_title = self.fig.suptitle(fig_title)

        for key, value in data_dict.items():
            self.axes.plot(value[:, 0], value[:, 1], color=class_color[key], label=key)

        # if data.ndim == 1:
        #     x = np.array(range(len(data)))
        #     self._data = np.c_[x, data]
        # else:
        #     self._data = data
        # if len(self._data.shape) == 3 and self._data.shape[2] == 2:
        #     for array in self._data:
        #         npoints = len(array)
        #         if plot_type=='scatter':
        #             self.axes.scatter(self._data[:, 0], self._data[:, 1], color=color, label=label)
        #         else:
        #             self.axes.plot(self._data[:, 0], self._data[:, 1], color=color, label=label)
        #
        # elif len(self._data.shape) == 2 and self._data.shape[1] == 2:
        #     npoints = len(self._data)
        #     if plot_type == 'scatter':
        #         self.axes.scatter(self._data[:, 0], self._data[:, 1], color=color, label=label)
        #     else:
        #         self.axes.plot(self._data[:, 0], self._data[:, 1], color=color, label=label)

        handles, labels = self.axes.get_legend_handles_labels()
        self.legend = self.axes.legend(handles, labels, loc=self.legend_loc)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.90)

    def save(self, path):
        if self.show_legend:
            self.fig.savefig(path, bbox_extra_artists=(self.legend,
                                                       self.x_axis_lbl,
                                                       self.y_axis_lbl,
                                                       self.fig_title,),
                             bbox_inches='tight')
        else:
            self.fig.savefig(path)


# ===================================================


class MyStaticMplCanvas(MplCanvas):
    """Example: Simple canvas with a sine plot."""

    def mpl_line_plot(self, data, x_label, y_label, color='r', label=None,
                      title='Plot',
                      use_grid=True,
                      use_hold=False):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MplCanvas):
    """Example: A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def mpl_line_plot(self, data, x_label, y_label, color='r', label=None,
                      title='Plot',
                      use_grid=True,
                      use_hold=False):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        import random
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]

        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()
