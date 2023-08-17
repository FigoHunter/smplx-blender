from smplx_blender import body,utils
import os

body_model=None


model_path=os.path.join(utils.DATA_PATH,"favor_preview","body_utils","body_models")

def getBodyModel(gender="male",device="cpu"):
    return body.get_body_model(model_path,"smplx", gender, 1, device=device)
