#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/dataloader/gait_video_dataset.py
Project: /workspace/skeleton/project/dataloader
Created Date: Monday December 11th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday December 11th 2023 11:26:34 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

27-03-2024	Kaixu Chen	make temporal mix a separate class.
"""

from __future__ import annotations

import logging, sys, json

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png
from pytorchvideo.transforms.functional import uniform_temporal_subsample

logger = logging.getLogger(__name__)

class_to_num_mapping_dict = {
    "left45_right45": 0,
    "left45_right90": 1,
    "left90_right45": 2,
    "left90_right90": 3,
}

class TemporalMix(object):
    """
    This class is temporal mix, which is used to mix the first phase and second phase of gait cycle.
    """

    def __init__(self, uniform_temporal_num=16) -> None:
        self.uniform_temporal_subsample = uniform_temporal_num

    def fuse_frames(
        self,
        first_phase: torch.Tensor,
        second_phase: torch.Tensor,
    ) -> torch.Tensor:

        # * fuse the frame with different phase
        uniform_first_phase = uniform_temporal_subsample(
            first_phase, self.uniform_temporal_subsample, temporal_dim=-4
        )  # t, c, h, w
        uniform_second_phase = uniform_temporal_subsample(
            second_phase, self.uniform_temporal_subsample, temporal_dim=-4
        )

        # fuse width dim
        fused_frames = torch.cat([uniform_first_phase, uniform_second_phase], dim=3)

        # write the fused frame to png
        for i in range(fused_frames.size()[0]):
            write_png(
                input=fused_frames[i],
                filename=f"/workspace/code/logs/img/fused{i}.png",
            )

        return fused_frames

    def __call__(
        self,
        video_tensor: torch.Tensor,
        left_index: int, 
        right_index: int,
    ) -> torch.Tensor:

        # * step1: first find the phase frames (pack) and phase index.
        first_phase = video_tensor[:right_index]
        second_phase = video_tensor[right_index:]

        # * step2: process on pack, fuse the frame
        fused_vframes = self.fuse_frames(first_phase, second_phase)

        return fused_vframes


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        video_path: str,
        video_index_path: str,
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        super().__init__()

        self._transform = transform
        self._experiment = experiment

        self._video_path = video_path
        self._video_index_path = video_index_path

        self.dataset_dict = self.prepare()

        if experiment == "temporal_mix":
            uniform_temporal_num = transform.transforms[0]._num_samples
            self._temporal_mix = TemporalMix(uniform_temporal_num)
        else:
            self._temporal_mix = False

    def prepare(
        self
    ):

        res = {}

        idx = 0
        for class_num in self._video_path.iterdir():
            for one_video_path in class_num.iterdir():
                video_name = one_video_path.name
                mapping_idx = self._video_index_path / video_name.replace(
                    ".mp4", ".json"
                )
                info_dict = {
                    "video_path": one_video_path,
                    "video_name": video_name,
                    "label": class_num.name,
                    "index": mapping_idx,
                }

                res[idx] = info_dict
                idx += 1

        return res

    def move_transform(self, vframes: torch.Tensor) -> None:

        if self._transform is not None:
            transformed_img = self._transform(vframes.permute(1, 0, 2, 3))
            return transformed_img  # c, t, h, w
        else:
            print("no transform")
            return vframes.permute(1, 0, 2, 3)  # c, t, h, w

    def __len__(self):

        return len(self.dataset_dict.keys())

    def __getitem__(self, index) -> Any:

        # unpackage video info
        video_path = self.dataset_dict[index]["video_path"]
        video_name = self.dataset_dict[index]["video_name"]
        video_mapping_index_path = self.dataset_dict[index]["index"]
        video_label = self.dataset_dict[index]["label"]

        with open(video_mapping_index_path, "r") as f:
            video_mapping = json.load(f)

        left_index = video_mapping["left_index"]
        right_index = video_mapping["right_index"]

        vframes, _, _ = read_video(video_path, output_format="TCHW")

        # TODO: 还有优化的空间
        if self._experiment == "temporal_mix":
            # should return the new frame, named temporal mix.
            defined_vframes = self._temporal_mix(vframes, left_index, right_index)
            defined_vframes = self.move_transform(defined_vframes)

        elif self._experiment == "late_fusion":

            stance_vframes = vframes[:right_index]
            swing_vframes = vframes[right_index:]

            trans_stance_vframes = self.move_transform(stance_vframes)
            trans_swing_vframes = self.move_transform(swing_vframes)

            # * 将不同的phase组合成一个batch返回
            defined_vframes = torch.stack(
                [trans_stance_vframes, trans_swing_vframes], dim=-1
            )

        elif "single" in self._experiment:
            if self._experiment == "single_stance":
                defined_vframes = vframes[:right_index]
            elif self._experiment == "single_swing":
                defined_vframes = vframes[right_index:]

            defined_vframes = self.move_transform(defined_vframes)

        else:
            raise ValueError("experiment name is not correct")

        sample_info_dict = {
            "video": defined_vframes,
            "label": class_to_num_mapping_dict[video_label],
            "video_name": video_name,
        }

        return sample_info_dict


def labeled_gait_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
    dataset_index_idx: Dict = None,
) -> LabeledGaitVideoDataset:

    dataset = LabeledGaitVideoDataset(
        experiment=experiment,
        video_path=dataset_idx,
        video_index_path=dataset_index_idx,
        transform=transform,
    )

    return dataset
