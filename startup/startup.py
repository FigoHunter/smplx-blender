import importlib
import os
from smplx_blender import utils

dir = utils.STARTUP_PATH

for f in os.listdir(dir):
    path = os.path.join(dir, f)
    if os.path.isdir(path):
        importlib.import_module(f)