import smplx

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