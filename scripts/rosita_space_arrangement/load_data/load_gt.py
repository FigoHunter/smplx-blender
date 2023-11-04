import os
import rosita_2023_08.load
from smplx_blender import utils


data_dir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1")

gt_path=os.path.join(data_dir,'A001-2023-0511-1802-23-task-236-seq-6.pkl')

gt_data=rosita_2023_08.load.loadGroundTruthData(gt_path)
grab_frame,end_frame=rosita_2023_08.load.loadGroundTruthMovingFrame(gt_data)
frames = gt_data["length"]
smplx_model=rosita_2023_08.smplx_model.getBodyModel('female')

for frame in range(frames):
    pass