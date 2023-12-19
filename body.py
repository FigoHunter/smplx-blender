import smplx

BODY_JOINT_NAMES = [
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]
BODY_JOINT_PARENT={
    'left_hip':'root',
    'right_hip': 'root',
    'spine1':'root',
    'left_knee':'left_hip',
    'right_knee':'right_hip',
    'spine2':'spine1',
    'left_ankle':'left_knee',
    'right_ankle':'right_knee',
    'spine3':'spine2',
    'left_foot':'left_ankle',
    'right_foot':'right_ankle',
    'neck':'spine3',
    'left_collar':'neck',
    'right_collar':'neck',
    'head':'neck',
    'left_shoulder':'left_collar',
    'right_shoulder':'right_collar',
    'left_elbow':'left_shoulder',
    'right_elbow':'right_shoulder',
    'left_wrist':'left_elbow',
    'right_wrist':'right_elbow',
}
FINGER_JOINT_NAMES = [
    'index1',
    'index2',
    'index3',
    'middle1',
    'middle2',
    'middle3',
    'pinky1',
    'pinky2',
    'pinky3',
    'ring1',
    'ring2',
    'ring3',
    'thumb1',
    'thumb2',
    'thumb3',
]
FINGER_JOINT_PARENT={
    'index1':'root',
    'index2':'index1',
    'index3':'index2',
    'middle1':'root',
    'middle2':'middle1',
    'middle3':'middle2',
    'pinky1':'root',
    'pinky2':'pinky1',
    'pinky3':'pinky2',
    'ring1':'root',
    'ring2':'ring1',
    'ring3':'ring2',
    'thumb1':'root',
    'thumb2':'thumb1',
    'thumb3':'thumb2',
}

def get_mirrored_joint(name:str):
    assert name.startswith('left') or name.startswith('right')
    if name.startswith('left'):
        return name.replace('left','right',1)
    else:
        return name.replace('right','left',1)

def get_body_model(path, type, gender, batch_size, device="cuda", v_template=None):
    """
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    """
    # ? you can download these from https://smpl-x.is.tue.mpg.de/index.html
    # ? please follow the instruction in SAGA https://github.com/JiahaoPlus/SAGA
    body_model_path = path
    body_model = smplx.create(
        body_model_path,
        model_type=type,
        gender=gender,
        ext="npz",
        use_pca=False,
        num_betas=100,
        num_pca_comps=45,
        flat_hand_mean=True,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=batch_size,
        v_template=v_template,
    )
    if device == "cuda":
        return body_model.cuda()
    else:
        return body_model