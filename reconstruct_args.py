from dataclasses import dataclass
from typing import Literal
import os

IMAGE_MODE = Literal["shuffle", "distributed", "mfps", "farthestpose", "nearestpose"]
COLMAP = os.path.expanduser("~/.conda/envs/vggt/bin/colmap")
CAMERA_TYPE = Literal["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]
COPY_MODE = Literal[None, "crop", "square", "tiles"]
COLMAP_MODE = Literal["default", "relaxed"]  # , "interpolated" TODO
SAMPLING_MODE = Literal["random", "confidence", "voxels", "none", "ba", "vox3", "fps", "imagefps"]


@dataclass
class ReconstructArgs:
    name: str
    input: str
    choice: str
    num_images: int | None
    seed: int
    conf_thres_value: float
    sampling_mode: SAMPLING_MODE
    image_mode: IMAGE_MODE
    camera_type: CAMERA_TYPE
    num_points: int
    colmap_mode: COLMAP_MODE = "default"
    copy_mode: COPY_MODE = None
    force: bool = False
    require_depth_conf: bool = False
    save_conf_as_errors: bool = False
    shared_camera: bool = False
    use_ba: bool = False
    max_ba_iterations: int = 50
    near_filtering_strength: float = 0.0
    near_filtering_quorum: int = 1
    reconstruct_pose_opt: bool = False
    optimisation_iterations: int = 0
    optimisation_neighbourhood: int = 10

    def __post_init__(self):
        self.name = self.name.strip("/")
        self.input = self.input

    @property
    def cache_name(self):
        name = self.name

        # Define cache directory based on params affecting raw predictions
        cache_parts = []
        if self.num_images:
            cache_parts.append(f"n{self.num_images}")
        cache_parts.append(f"s{self.seed}")
        cache_parts.append(self.image_mode)

        if self.choice == "colmap":
            if self.shared_camera:
                cache_parts.append("sharedcam")

        if self.copy_mode is not None:
            cache_parts.append(self.copy_mode)

        cache_name = name + "_cache_" + "_".join(cache_parts)
        return cache_name

    @property
    def cache_dir(self):
        cache_dir = f"./{self.choice}_outputs/{self.cache_name}"
        return cache_dir

    @property
    def full_name(self):
        parts = []
        name = self.name

        if self.num_images:
            parts.append(f"n{self.num_images}")

        parts.append(f"s{self.seed}")

        if self.choice == "vggt":
            if self.sampling_mode == "ba" or self.use_ba:
                parts.append(self.camera_type.lower().replace("simple_", "m"))
            if self.sampling_mode != "ba":
                parts.extend(
                    [
                        f"c{self.conf_thres_value}",
                        f"p{self.num_points}",
                    ]
                )
            parts.append(self.sampling_mode)
            if self.near_filtering_strength > 0.0:
                parts.append(f"nf{self.near_filtering_strength}")
                parts.append(f"nq{self.near_filtering_quorum}")

            if self.reconstruct_pose_opt:
                parts.append("recposeopt")

            if self.optimisation_iterations > 0:
                parts.append(f"opti{self.optimisation_iterations}")
                parts.append(f"optn{self.optimisation_neighbourhood}")

            if self.save_conf_as_errors:
                parts.append("errconf")
        elif self.choice == "colmap":
            parts.append(self.colmap_mode)
            if self.camera_type != "SIMPLE_RADIAL":
                parts.append(self.camera_type.lower().replace("simple_", "m"))

        parts.append(self.image_mode)

        if self.use_ba and self.choice != "colmap":
            parts.append("useba")

        if self.choice == "vggt" and (self.sampling_mode == "ba" or self.use_ba):
            parts.append(f"maxba{self.max_ba_iterations}")

        if self.choice == "colmap" or self.sampling_mode == "ba" or self.use_ba:
            if self.shared_camera:
                parts.append("sharedcam")

        if self.copy_mode is not None:
            parts.append(self.copy_mode)

        # name = f"{name}_s{seed}_c{conf_thres_value}_p{num_points}_{sampling_mode}"
        name = name + "_" + "_".join(parts)
        return name

    @property
    def base_out(self):
        base_out = f"./{self.choice}_outputs/{self.full_name}"
        return base_out
