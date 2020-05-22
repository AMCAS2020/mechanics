import numpy as np

def astype_float32(input_function):
    def entry_exit(*args):
        cnt = input_function(*args)

        return cnt.astype(np.float32)

    return entry_exit


def to_contour_opencvformat(input_function):
    def entry_exit(*args):
        result_cnt = input_function(*args)
        if len(np.array(result_cnt).shape) == 1:
            contour = np.array(result_cnt).reshape((-1, 1, 2))
        elif np.array(result_cnt).shape[2] == 2:
            contour = np.array(result_cnt).reshape((-1, 1, 2))
        elif np.array(result_cnt).shape[2] == 3:
            contour = np.array(result_cnt).reshape((-1, 1, 3))

        contour_opencvformat = [np.asarray(contour)]
        return contour_opencvformat

    return entry_exit


def landmark_2d_output_format(input_function):
    def entry_exit(*args, **kwargs):
        result = input_function(*args, **kwargs)

        res_shape = np.array(result).shape[-1]

        if res_shape != 0:
            result = np.asarray(result, dtype=np.float32).reshape(-1, 1, res_shape)
        # print("result=",result)
        return result

    return entry_exit


def input_point_to_2D_numpy_array(input_function):
    def entry(*args):
        new_args = []
#        print("args =", args)
        for i in range(len(args)):

            if len(args[i]) == 3:
#                print("args =", args[i])
                new_args.append(np.array([[args[i][0], args[i][1]]]))
#                print(tuple(new_args))
            elif i == 1:
                new_args.append(args[i])
            else:
                raise ValueError("please insert points as a list or tuple "
                                 "e.x: [x, y, z] or (x, y, z)")
        result = input_function(*tuple(new_args))

        return result

    return entry


def dict_output(input_function):
    def output(*args):
        result = input_function(*args)
        if not isinstance(result, dict):
            raise TypeError("return value is not type dict!")

        return result

    return output

