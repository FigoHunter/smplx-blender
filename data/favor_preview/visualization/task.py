import argparse
import glob
import json
import os
import pickle
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import trimesh

from visualization.transform import generate_rand_transf, aa_to_rotmat


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return obj.tolist()
        else:
            return [convert_numpy(row) for row in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


class ObjectMesh:
    def __init__(self) -> None:
        self.source = None
        self.oid = None
        self.obj_path = None  # ! MUST use the relative path

    def to_dict(self):
        return {
            "source": self.source,
            "oid": self.oid,
            "obj_path": self.obj_path,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.source = d["source"]
        obj.oid = d["oid"]
        obj.obj_path = d["obj_path"]
        return obj


class IndicatorObject:
    def __init__(self):
        self.type = None
        self.size = []  # should be a list of 2/3 floats, [l, w, h] or [r, h]
        self.color = ""
        self.color_dict = {
            "red": [1, 0, 0],
            "green": [0, 1, 0],
            "blue": [0, 0, 1],
            "yellow": [1, 1, 0],
            "purple": [1, 0, 1],
            "cyan": [0, 1, 1],
        }

    def to_dict(self):
        return {
            "type": self.type,
            "size": self.size,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.type = d["type"]
        obj.size = d["size"]
        obj.color = d["color"]
        return obj

    def __str__(self) -> str:
        return f"the {self.color} {self.type}"

    @classmethod
    def random_generate(cls, color_idx=None, xy_bounds=[0.05, 0.3], h_bounds=[0.05, 0.3]):
        indicator = cls()
        indicator.type = random.choice(["cube", "cylinder"])
        color_list = ["red", "green", "blue", "yellow", "purple", "cyan"]
        if color_idx is not None:
            indicator.color = color_list[color_idx]
        else:
            indicator.color = random.choice(color_list)
        if indicator.type == "cube":
            indicator.size = [random.uniform(*xy_bounds), random.uniform(*xy_bounds), random.uniform(*h_bounds)]
        elif indicator.type == "cylinder":
            indicator.size = [random.uniform(xy_bounds[0] / 2, xy_bounds[1] / 2), random.uniform(*h_bounds)]
        else:
            raise NotImplementedError
        return indicator

    def to_o3d(self):
        import open3d as o3d

        if self.type == "cube":
            o3d_obj = o3d.geometry.TriangleMesh.create_box(width=self.size[0], height=self.size[1], depth=self.size[2])
        elif self.type == "cylinder":
            o3d_obj = o3d.geometry.TriangleMesh.create_cylinder(radius=self.size[0], height=self.size[1])
        else:
            raise NotImplementedError
        o3d_obj.paint_uniform_color(self.color_dict[self.color])
        return o3d_obj


class OakInkObjectMesh(ObjectMesh):
    def __init__(self) -> None:
        super().__init__()
        self.source = "OakInk"

    @classmethod
    def get_obj_path(cls, oid, data_path, use_downsample=True, key="align"):
        obj_suffix_path = "align_ds" if use_downsample else "align"
        real_meta = json.load(open(os.path.join(data_path, "metaV2/object_id.json"), "r"))
        virtual_meta = json.load(open(os.path.join(data_path, "metaV2/virtual_object_id.json"), "r"))
        if oid in real_meta:
            obj_name = real_meta[oid]["name"]
            obj_path = os.path.join(data_path, "OakInkObjectsV2")
        else:
            obj_name = virtual_meta[oid]["name"]
            obj_path = os.path.join(data_path, "OakInkVirtualObjectsV2")
        obj_mesh_path = list(
            glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj"))
            + glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply"))
        )
        if len(obj_mesh_path) > 1:
            obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
        assert len(obj_mesh_path) == 1
        return obj_mesh_path[0]

    @classmethod
    def generate(cls, idx):
        obj = cls()
        obj_dict = json.load(open(f"data/OakInk/shape/metaV2/object_id.json"))
        obj_dict.update(json.load(open(f"data/OakInk/shape/metaV2/virtual_object_id.json")))
        oid = sorted(obj_dict.keys())

        obj.oid = oid[idx]
        obj.obj_path = cls.get_obj_path(obj.oid, "data/OakInk/shape", use_downsample=True)

        return obj

    def get_mujoco_xml_path(self):
        xml_path = self.obj_path.replace("data/OakInk/shape/", "tmp/mjcf_oakink/")
        file_name = os.path.split(self.obj_path)[1].split(".")[0]
        xml_path = (
            xml_path.replace("/align_ds/", f"/{file_name}/")
            .replace("/align_ds/", f"/{file_name}/")
            .replace(".obj", ".xml")
            .replace(".ply", ".xml")
        )
        return xml_path


class Task(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = self.__class__.__name__

    @classmethod
    def random_transf(cls, obj_verts, x_bounds, y_bounds, z_bounds):
        while True:
            transf = generate_rand_transf(1, r_angle=np.pi, r_tsl=1.0)[0].numpy()  # [4, 4]
            transf[0, 3] = transf[0, 3] * ((x_bounds[1] - x_bounds[0]) / 2) + (x_bounds[1] + x_bounds[0]) / 2
            transf[1, 3] = transf[1, 3] * ((y_bounds[1] - y_bounds[0]) / 2) + (y_bounds[1] + y_bounds[0]) / 2
            transf[2, 3] = transf[2, 3] * ((z_bounds[1] - z_bounds[0]) / 2) + (z_bounds[1] + z_bounds[0]) / 2
            obj_verts_transf = transf[:3, :3] @ obj_verts.T + transf[:3, [3]]  # [3, N]
            if np.all(obj_verts_transf[1] > 0):  # y > 0
                break
        return transf

    @classmethod
    def from_dict(cls, d):
        task = cls()
        if d["task_name"] == "move":
            task = MoveTask.from_dict(d)
        elif d["task_name"] == "move_to":
            task = MoveToTask.from_dict(d)
        else:
            raise NotImplementedError
        return task


class MoveTask(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "move"
        self.object_mesh = ObjectMesh()
        self.init_pose = None
        self.target_pose = None

    @classmethod
    def random_generate(
        cls, random_id=None, x_bounds=(-0.6, 0.6), y_bounds=(0.0, 0.6), z_bounds=(-0.3, 0.3), mujoco_check=False
    ):
        task = cls()
        # SEED = 8
        # random.seed(SEED)
        # np.random.seed(SEED)
        # torch.manual_seed(SEED)
        if random_id is None:
            random_id = random.randint(0, 1800)
        task.object_mesh = OakInkObjectMesh.generate(random_id)
        obj_verts = trimesh.load(task.object_mesh.obj_path, process=False, force="mesh", skip_materials=True).vertices
        task.init_pose = cls.random_transf(obj_verts, x_bounds, y_bounds, z_bounds)
        task.target_pose = cls.random_transf(obj_verts, x_bounds, y_bounds, z_bounds)
        if mujoco_check:
            xml_path = task.object_mesh.get_mujoco_xml_path()
            task.init_pose, _ = simulate(xml_path, init_pose=task.init_pose, vis=False)

        return task

    def to_dict(self):
        return {
            "task_name": self.task_name,
            "object_mesh": self.object_mesh.to_dict(),
            "init_pose": convert_numpy(self.init_pose),
            "target_pose": convert_numpy(self.target_pose),
        }

    @classmethod
    def from_dict(cls, d):
        task = cls()
        task.task_name = d["task_name"]
        if d["object_mesh"]["source"] == "OakInk":
            task.object_mesh = OakInkObjectMesh.from_dict(d["object_mesh"])
        else:
            task.object_mesh = ObjectMesh.from_dict(d["object_mesh"])
        task.init_pose = np.array(d["init_pose"])
        task.target_pose = np.array(d["target_pose"])
        return task


class MoveToTask(MoveTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "move_to"
        self.indicator_objects = []
        self.preposition = None
        self.key_indicator = 0

    @classmethod
    def collision(cls, v1, v2):
        x_range_1 = (np.min(v1[:, 0]), np.max(v1[:, 0]))
        z_range_1 = (np.min(v1[:, 2]), np.max(v1[:, 2]))
        x_range_2 = (np.min(v2[:, 0]), np.max(v2[:, 0]))
        z_range_2 = (np.min(v2[:, 2]), np.max(v2[:, 2]))

        # from termcolor import cprint

        # cprint(x_range_1, "green")
        # cprint(x_range_2, "red")
        # cprint(z_range_1, "blue")
        # cprint(z_range_2, "yellow")

        if x_range_1[1] < x_range_2[0] or x_range_1[0] > x_range_2[1]:
            return False
        if z_range_1[1] < z_range_2[0] or z_range_1[0] > z_range_2[1]:
            return False
        return True

    @classmethod
    def random_generate(
        cls, random_id=None, x_bounds=(-0.6, 0.6), y_bounds=(0.0, 0.6), z_bounds=(-0.3, 0.3), mujoco_check=False
    ):
        task = super().random_generate(random_id, x_bounds, y_bounds, z_bounds, mujoco_check)
        task.task_name = "move_to"
        task.indicator_objects = []
        task.preposition = random.choices(
            [
                "on top of",
                "on the left of",
                "on the right of",
                "in front of",
                "behind",
            ],
            [0.6, 0.1, 0.1, 0.1, 0.1],
            k=1,
        )[0]
        indicator_num = random.randint(1, 4)
        task.key_indicator = random.randint(0, indicator_num - 1)
        for _ in range(indicator_num):
            while True:
                indicator_object = IndicatorObject.random_generate(color_idx=len(task.indicator_objects))
                height = indicator_object.size[1]
                rand_transf = np.eye(4)
                rand_transf[:3, :3] = aa_to_rotmat(np.array([0, random.uniform(-np.pi, np.pi), 0])) @ aa_to_rotmat(
                    np.array([-np.pi / 2, 0, 0])
                )
                try:
                    if indicator_object.type == "cylinder":
                        xz_expand = max(indicator_object.size[0] - 0.1, 0.0)
                    else:
                        xz_expand = max(max(indicator_object.size[0], indicator_object.size[2]) - 0.1, 0.0)

                    rand_transf[0, 3] = random.uniform(x_bounds[0] + xz_expand, x_bounds[1] - xz_expand)
                    if indicator_object.type == "cylinder":
                        rand_transf[1, 3] = height / 2
                    rand_transf[2, 3] = random.uniform(z_bounds[0] + xz_expand, z_bounds[1] - xz_expand)
                except:
                    continue

                indicator_obj_verts = np.asfarray(indicator_object.to_o3d().vertices)
                indicator_obj_verts = rand_transf[:3, :3] @ indicator_obj_verts.T + rand_transf[:3, [3]]
                # check collision with object
                obj_verts = trimesh.load(
                    task.object_mesh.obj_path, process=False, force="mesh", skip_materials=True
                ).vertices
                obj_verts = task.init_pose[:3, :3] @ obj_verts.T + task.init_pose[:3, [3]]
                collision = False
                collision = collision or cls.collision(obj_verts.T, indicator_obj_verts.T)

                # from termcolor import cprint

                # cprint(indicator_object.size, "green")
                # check collision with other indicator objects
                for pre_indicator_object in task.indicator_objects:
                    pre_indicator_obj_verts = np.asfarray(pre_indicator_object["obj"].to_o3d().vertices)
                    pre_indicator_obj_verts = (
                        pre_indicator_object["pose"][:3, :3] @ pre_indicator_obj_verts.T
                        + pre_indicator_object["pose"][:3, [3]]
                    )
                    collision = collision or cls.collision(pre_indicator_obj_verts.T, indicator_obj_verts.T)
                if not collision:
                    task.indicator_objects.append({"obj": indicator_object, "pose": rand_transf})
                    break
        return task

    def to_dict(self):
        d = super().to_dict()
        d["indicator_objects"] = [
            {"obj": obj["obj"].to_dict(), "pose": convert_numpy(obj["pose"])} for obj in self.indicator_objects
        ]
        d["preposition"] = self.preposition
        d["key_indicator"] = self.key_indicator
        return d

    @classmethod
    def from_dict(cls, d):
        task = super().from_dict(d)
        task.indicator_objects = [
            {"obj": IndicatorObject.from_dict(obj["obj"]), "pose": np.array(obj["pose"])}
            for obj in d["indicator_objects"]
        ]
        task.preposition = d["preposition"]
        task.key_indicator = d["key_indicator"]
        return task
