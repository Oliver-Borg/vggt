from dataclasses import dataclass
from typing import Literal
import os

IMAGE_MODE = Literal[
    "shuffle", "distributed", "mfps", "farthestpose", "minfarthestpose", "nearestpose", "minnearestpose"
]
COLMAP = os.path.expanduser("~/.conda/envs/vggt/bin/colmap")
CAMERA_TYPE = Literal["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]
COPY_MODE = Literal[None, "crop", "square", "tiles"]
COLMAP_MODE = Literal["default", "relaxed"]  # , "interpolated" TODO
SAMPLING_MODE = Literal[
    "random",
    "confidence",
    "voxels",
    "none",
    "ba",
    "vox3",
    "fps",
    "imagefps",
    "imagefreq",
    "imagefreqnovis",
    "imagefreqvoxels",
]
FEATURE_EXTRACTOR = Literal["sift", "sp", "aliked", "aliked+sp", "aliked+sp+sift"]
RECON_MODEL = Literal["vggt", "omega"]
OMEGA_CHECKPOINT = Literal["512", "416"]
# (checkpoint file, image_resolution of the preprocessing) of the VGGT-Omega checkpoints; 416 is the retrained
# "reproduce" checkpoint of the VGGT-Omega repository (reproduction.md), 512 the original release
OMEGA_CHECKPOINTS = {
    "512": ("~/work/checkpoints/vggt-omega/vggt_omega_1b_512.pt", 512),
    "416": ("~/work/checkpoints/vggt-omega/vggt_omega_1b_416_reproduce.pt", 416),
}


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
    feature_extractor: FEATURE_EXTRACTOR = "aliked+sp"
    run_mvs: bool = False
    enable_timing: bool = False
    save_full_pcd: bool = False
    model: RECON_MODEL = "vggt"
    omega_checkpoint: OMEGA_CHECKPOINT = "512"

    def __post_init__(self):
        self.name = self.name.strip("/")
        self.input = self.input
        if self.model == "omega":
            # Omega only predicts cameras and depth: no tracks, so no BA
            assert self.choice == "vggt", "model=omega is a backend of choice=vggt"
            assert not self.use_ba and self.sampling_mode != "ba", "Omega has no track head, so no BA"
            assert not self.reconstruct_pose_opt and self.optimisation_iterations == 0

    @property
    def model_token(self) -> str | None:
        """Run-name token of the Omega backend (None for VGGT, whose names are unchanged)."""
        if self.model != "omega" or self.choice != "vggt":
            return None
        return "omega" if self.omega_checkpoint == "512" else f"omega{self.omega_checkpoint}"

    @property
    def cache_name(self):
        name = self.name

        # Define cache directory based on params affecting raw predictions
        cache_parts = []
        if self.num_images:
            cache_parts.append(f"n{self.num_images}")
        cache_parts.append(f"s{self.seed}")
        if self.model_token:
            cache_parts.append(self.model_token)
        cache_parts.append(self.image_mode)

        if self.choice == "colmap":
            if self.shared_camera:
                cache_parts.append("sharedcam")
            if self.run_mvs:
                cache_parts.append("mvs")

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

        if self.model_token:
            parts.append(self.model_token)

        if self.choice == "vggt":
            if self.sampling_mode == "ba" or self.use_ba:
                parts.append(self.camera_type.lower().replace("simple_", "m"))
                parts.append(self.feature_extractor.replace("+", "-"))
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

            if self.enable_timing:
                parts.append("timing")

            if self.save_full_pcd:
                parts.append("fullpcd")

        elif self.choice == "colmap":
            parts.append(self.colmap_mode)
            if self.camera_type != "SIMPLE_RADIAL":
                parts.append(self.camera_type.lower().replace("simple_", "m"))
            if self.run_mvs:
                parts.append("mvs")

        parts.append(self.image_mode)

        if self.use_ba and self.choice != "colmap":
            parts.append("useba")

        if self.choice == "vggt" and (self.sampling_mode == "ba" or self.use_ba):
            parts.append(f"maxba{self.max_ba_iterations}")

        # if self.choice == "colmap" or self.sampling_mode == "ba" or self.use_ba:
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
