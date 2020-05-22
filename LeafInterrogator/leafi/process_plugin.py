import ast
import importlib
import inspect
import os


class Plugin:
    def __init__(self, plugins_directory):
        self._directory = plugins_directory
        self.process_dict = {}

    def load_plugins(self):
        """
        This method prepare a mapping between parent name and the it's
        children, which we used to build QTreeWidget.
        :return: Dictionary which the key is parent name and the value is
        list of child processes
        """
        self.process_dict = self.get_all_module_class()

        loaded_dict = self.load_classes()

        loaded_classes = [value for value in loaded_dict.values()]

        if not self.process_dict:
            print("Unable to found classes in: {}".format(self._directory))
        result_tree_dict = {}
        for cl in loaded_classes:
            # if not issubclass(cl, ABC):
            #     continue
            try:
                pr_list = result_tree_dict[cl.parent_name]
                pr_list.append(cl.process_name)
                result_tree_dict[cl.parent_name] = pr_list
            except KeyError:
                result_tree_dict[cl.parent_name] = [cl.process_name]

        return result_tree_dict

    def get_all_module_class(self):
        """
        This method find classes in a module
        :return: Dictionary which key is module name and the
        value is list of all classes in that module.
        """
        result_dict = {}
        for data in os.listdir(self._directory):
            if data != "__init__.py" and data != '__pycache__':
                with open(os.path.join(self._directory, data), 'r') as f:
                    p = ast.parse(f.read())

                classes_names = [node.name for node in ast.walk(p) if
                                 isinstance(node, ast.ClassDef)]
                module_name = data.split('.')[0]

                result_dict[module_name] = classes_names

        return result_dict

    def load_classes(self):
        """
        This method load all classes and map class name to the
        corresponding loaded class.
        :return: Dictionary of class name and loaded class object.
        """
        loaded_dict = {}
        for module, class_list in self.process_dict.items():
            dir = self._directory.split('/')[1] + '.' + \
                  self._directory.split('/')[2] + '.' + module
            for class_name in class_list:
                if class_name == 'Helper':
                    continue
                imp_mod = importlib.import_module(dir)
                cls = getattr(imp_mod, class_name)
                cls_obj = cls()
                loaded_dict[class_name] = cls_obj
                # s = getattr(imp_mod, class_name)
                # # print("signature=", inspect.signature(s.__init__))
                # # print(inspect.getmembers(s))
                # # print(inspect.)

        return loaded_dict

    def create_dict_processname_and_classname(self):
        """
        This function map process name to the class name
        :return: Dictionary which the key is process name and the value is
        class name.
        """
        self.process_dict = self.get_all_module_class()
        loaded_dict = self.load_classes()

        processname_and_classname_dict = {}
        for classname, loadedclass in loaded_dict.items():
            # if not issubclass(loadedclass, ABC):
            #     continue
            processname_and_classname_dict[loadedclass.process_name] = \
                classname

        return processname_and_classname_dict

    def get_all_class_parameters(self, selected_class):
        attributes = inspect.getmembers(selected_class, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not ((a[0].startswith('__') and
                                                     a[0].endswith('__')) or
                                                    a[0].startswith('_'))]
        # attributes_1 = vars(selected_class)
        return attributes
