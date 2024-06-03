#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/dataloader/data_loader.py
Project: /workspace/code/project/dataloader
Created Date: Sunday June 2nd 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Sunday June 2nd 2024 2:03:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from torchvision.transforms import (
    Compose,
    Resize,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Div255,
)
from customize_transforms import (
    AddSaltAndPepperNoise,
    AddGaussianNoise,
    AddUniformNoise,
)

from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from video_dataset import labeled_gait_video_dataset

disease_to_num_mapping_Dict: Dict = {
    4: {"left45_right45": 0, "left45_right90": 1, "left90_right45": 2, "left90_right90": 3},
}

class PendulumDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._data_path = opt.data.data_path
        self._mapping_index_path = opt.data.index_path
        self._config_path = opt.data.config_path

        # ? 感觉batch size对最后的结果有影响，所以分开使用不同的batch size
        self._gait_cycle_batch_size = opt.data.gait_cycle_batch_size
        self._default_batch_size = opt.data.default_batch_size

        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        # frame rate
        self._CLIP_DURATION = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx
        self._class_num = opt.model.model_class_num

        self._experiment = opt.train.experiment

        if self._experiment == "temporal_mix":
            self.mapping_transform = Compose(
                [Div255(), Resize(size=[self._IMG_SIZE, self._IMG_SIZE])]
            )
        else:
            self.mapping_transform = Compose(
                [
                    UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                    Div255(),
                    Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                ]
            )

        self.train_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

        self.val_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        if self._experiment == "temporal_mix":

            # train dataset
            self.train_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    0
                ],  # train mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

            # val dataset
            self.val_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

            # test dataset
            self.test_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

        elif "single" in self._experiment:

            # train dataset
            if self._experiment == "single_random":
                self.train_gait_dataset = labeled_video_dataset(
                    data_path=self._dataset_idx[2],
                    clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                    transform=self.train_video_transform,
                )

            else:
                self.train_gait_dataset = labeled_gait_video_dataset(
                    experiment=self._experiment,
                    dataset_idx=self._dataset_idx[
                        0
                    ],  # train mapped path, include gait cycle index.
                    transform=self.mapping_transform,
                )

            # val dataset
            self.val_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_video_transform,
            )

            # test dataset
            self.test_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_video_transform,
            )

        elif self._experiment == "late_fusion":

            # train dataset
            self.train_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    0
                ],  # train mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )
            
            # val dataset
            self.val_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

            # test dataset
            self.test_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

        else:
            raise ValueError("experiment keyword not support")

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """

        batch_label = []
        batch_video = []

        # * mapping label
        for i in batch:
            # logging.info(i['video'].shape)
            gait_num, *_= i["video"].shape
            disease = i["disease"]

            batch_video.append(i["video"])
            for _ in range(gait_num):

                if disease in disease_to_num_mapping_Dict[self._class_num].keys():

                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num][disease]
                    )
                else:
                    # * if the disease not in the mapping dict, then set the label to non-ASD.
                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num]["non-ASD"]
                    )

        # video, b, c, t, h, w, which include the video frame from sample info
        # label, b, which include the video frame from sample info
        # sample info, the raw sample info from dataset
        return {
            "video": torch.cat(batch_video, dim=0),
            "label": torch.tensor(batch_label),
            "info": batch,
        }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        if self._experiment == "single_random":
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )
        else:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._gait_cycle_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        
        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        if "single" in self._experiment:
            val_data_loader = DataLoader(
                self.val_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )
        else:
            val_data_loader = DataLoader(
                self.val_gait_dataset,
                # batch_size=self._gait_cycle_batch_size,
                batch_size=16,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        
        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        
        if "single" in self._experiment:
            val_data_loader = DataLoader(
                self.val_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )
        else:
            val_data_loader = DataLoader(
                self.val_gait_dataset,
                # batch_size=self._gait_cycle_batch_size,
                batch_size = 16,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        
        return val_data_loader