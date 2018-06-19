## pylint: disable=C0103, C0111
from os import listdir
from os.path import join
import numpy as np
from noformat import File


def convert_data(data_path: str):
    data = File(data_path, 'w+')
    for x in data:
        if isinstance(data[x], np.lib.npyio.NpzFile):
            array = data[x]
            if 'x' in array:
                continue
            if 'column' in array:
                if 'z' in array:
                    data[x] = {'x': array['row'], 'y': array['column'], 'data': array['data'], 'z': array['z']}
                else:
                    data[x] = {'x': array['row'], 'y': array['column'], 'data': array['data']}


def convert_folder(folder_path: str):
    for folder in listdir(folder_path):
        convert_data(join(folder_path, folder))
##
