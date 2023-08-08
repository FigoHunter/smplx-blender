import os
import numpy as np

HOME_PATH=os.getenv("HOME")
DATA_PATH=os.path.join(HOME_PATH,"data")


class ndarray_pydata(np.ndarray):
    def __bool__(self) -> bool:
        return len(self) > 0

    @staticmethod
    def parse(array:np.ndarray):
        return array.view(ndarray_pydata)