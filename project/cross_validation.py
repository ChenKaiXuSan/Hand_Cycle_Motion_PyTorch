#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/cross_validation.py
Project: /workspace/code/project
Created Date: Monday June 3rd 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday June 3rd 2024 7:28:54 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import os, json, shutil, copy, random
from typing import Any, Dict, List, Tuple

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import StratifiedGroupKFold, train_test_split, GroupKFold, KFold
from pathlib import Path

# * this used for loading the mapping info.
class_num_mapping_Dict: Dict = {
    4: {
        0: "left45_right45",
        1: "left45_right90",
        2: "left90_right45",
        3: "left90_right90",
    }
}

class DefineCrossValidation(object):
    """process:
    cross validation > over/under sampler > train/val split
    fold: [train/val]: [path]
    """

    def __init__(self, config) -> None:
        
        self.video_path: Path = Path(config.data.data_path)
        self.index_path: Path = Path(config.data.index_path)
        # self.gait_seg_idx_path: Path = Path(config.data.gait_seg_index_data_path) # used for training path mapping

        self.K: int = config.train.fold
        # self.sampler: str = config.data.sampling # data balance, [over, under, none]

        self.class_num: int = config.model.model_class_num

    @staticmethod
    def random_sampler(X: list, y: list, train_idx: list, val_idx: list, sampler):
        # train
        train_mapped_path = []
        new_X_path = [X[i] for i in train_idx]

        sampled_X, sampled_y = sampler.fit_resample(
            [[i] for i in range(len(new_X_path))], [y[i] for i in train_idx]
        )

        # map sampled_X to new_X_path
        for i in sampled_X:
            train_mapped_path.append(new_X_path[i[0]])

        # val
        val_mapped_path = []
        new_X_path = [X[i] for i in val_idx]

        sampled_X, sampled_y = sampler.fit_resample(
            [[i] for i in range(len(new_X_path))], [y[i] for i in val_idx]
        )

        # map
        for i in sampled_X:
            val_mapped_path.append(new_X_path[i[0]])

        return train_mapped_path, val_mapped_path

    def process_cross_validation(self, video_dict: dict) -> Tuple[List, List, List]:

        _path = video_dict

        X = []  # patient index
        y = []  # patient class index
        groups = []  # different patient groups

        disease_to_num = {disease: idx for idx, disease in class_num_mapping_Dict[self.class_num].items()}
        element_to_num = {}

        name_map = set()

        # process one disease in one loop.
        for disease, path in _path.items():
            patient_list = sorted(list(path))

            for p in patient_list:
                name, _ = p.name.split("-")
                #  FIXME: 我觉得HipOA是造成数据不平衡的原因，所以我把HipOA的数据去掉了
                if 'HipOA' not in name:
                    name_map.add(name)

        for idx, element in enumerate(name_map):
            element_to_num[element] = idx
        
        for disease, path in _path.items():
            patient_list = sorted(list(path))
            for i in range(len(patient_list)):

                name, _ = patient_list[i].name.split("-")
                
                label = disease_to_num[disease]

                # FIXME: 我举得HipOA是造成数据不平衡的原因，所以我把HipOA的数据去掉了
                if 'HipOA' not in name:
                    X.append(patient_list[i])  # true path in Path
                    y.append(label)  # label, 0, 1, 2
                    groups.append(element_to_num[name])  # number of different patient

        return X, y, groups

    def make_dataset_with_video(self, val_dataset_idx: list, fold: int, flag: str):
        temp_path = self.gait_seg_idx_path / str(self.class_num) / self.sampler / str(fold) / str(flag)
        val_idx = val_dataset_idx

        _class_map = class_num_mapping_Dict[self.class_num]
        _disease_to_num = {disease: idx for idx, disease in _class_map.items()}

        shutil.rmtree(temp_path, ignore_errors=True)

        for path in val_idx:
            with open(path) as f:
                file_info_dict = json.load(f)

            video_name = file_info_dict["video_name"]
            video_path = file_info_dict["video_path"]
            video_disease = file_info_dict["disease"]

            if video_disease not in _disease_to_num.keys():
                video_disease = "non-ASD"

            if not (temp_path / video_disease).exists():
                (temp_path / video_disease).mkdir(parents=True, exist_ok=False)

            shutil.copy(
                video_path, temp_path / video_disease / (video_name + ".mp4")
            )

        return temp_path

    @staticmethod
    def magic_move(train_mapped_path, val_mapped_path):

        new_train_mapped_path = copy.deepcopy(train_mapped_path)
        new_val_mapped_path = copy.deepcopy(val_mapped_path)

        # train magic 
        train_tmp_dict = {}
        for i in train_mapped_path:
            # not move ASD
            if 'ASD' in i.name: continue;

            train_tmp_dict[i.name.split("-")[0]] = i

        val_tmp_dict = {}
        for i in val_mapped_path:
            # not move ASD 
            if 'ASD' in i.name: continue;
            val_tmp_dict[i.name.split("-")[0]] = i

        for k, v in train_tmp_dict.items():
            new_val_mapped_path.append(v)

            rm_idx = new_train_mapped_path.index(v)
            new_train_mapped_path.pop(rm_idx)

        for k, v in val_tmp_dict.items():
            new_train_mapped_path.append(v)

            rm_idx = new_val_mapped_path.index(v)
            new_val_mapped_path.pop(rm_idx)

        return new_train_mapped_path, new_val_mapped_path
    
    @staticmethod
    def map_class_num(class_num: int, raw_video_path: Path) -> Dict:

        _class_num = class_num_mapping_Dict[class_num]

        res_dict = {v:[] for k,v in _class_num.items()}

        for train_flag in raw_video_path.iterdir():

            for one_class in train_flag.iterdir():
                
                for one_video in one_class.iterdir():
                    res_dict[one_class.name].append(one_video)
        # TODO: 这里按照类别统计了所有的video。但是在cross validation的时候，是需要按照一个类别来划分的吗
        # TODO：划分的时候按照X，y组成一个pair就行了，不需要按照类别来划分。也就是说需要把他们都组合起来。
        # 但是需要前提确保不同类别中的顺序是保持的才行。
        return res_dict


    def prepare(self):
        """define cross validation first, with the K.
        #! the 1 fold and K fold should return the same format.
        fold: [train/val]: [path]

        Args:
            video_path (str): the index of the video path, in .json format.
            K (int, optional): crossed number of validation. Defaults to 5, can be 1 or K.

        Returns:
            list: the format like upper.
        """
        K = self.K

        ans_fold = {}

        mapped_class_Dict = self.map_class_num(self.class_num, self.video_path)

        # define the cross validation
        # X: video path, in path.Path foramt. len = 1954
        # y: label, in list format. len = 1954, type defined by class_num_mapping_Dict.
        # groups: different patient, in list format. It means unique patient index. [54]
        X, y, groups = self.process_cross_validation(mapped_class_Dict)

        sgkf = StratifiedGroupKFold(n_splits=K)

        for i, (train_index, test_index) in enumerate(
            sgkf.split(X=X, y=y, groups=groups)
        ):
            if self.sampler in ["over", "under"]:
                if self.sampler == "over":
                    ros = RandomOverSampler(random_state=42)
                elif self.sampler == "under":
                    ros = RandomUnderSampler(random_state=42)

                train_mapped_path, val_mapped_path = self.random_sampler(
                    X, y, train_index, test_index, ros
                )

            else:
                train_mapped_path = [X[i] for i in train_index]
                val_mapped_path = [X[i] for i in test_index]

            # FIXME: magic move 
            train_mapped_path, val_mapped_path = self.magic_move(train_mapped_path, val_mapped_path)

            # make the val data path
            train_video_path = self.make_dataset_with_video(train_mapped_path, i, "train")
            val_video_path = self.make_dataset_with_video(val_mapped_path, i, "val")

            # * here used for gait labeled method, or load video from path
            ans_fold[i] = [train_mapped_path, val_mapped_path, train_video_path, val_video_path]

        return ans_fold, X, y, groups

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        target_path = self.video_path
        index_path = self.index_path

        # * when json file changed, need to reprocess the dataset.
        if os.path.exists(target_path):
            
            # * step 1: get total data
            fold_dataset_idx, *_ = self.prepare()

            json_fold_dataset_idx = copy.deepcopy(fold_dataset_idx)

            for k, v in fold_dataset_idx.items():
                
                # train mapping path, include the gait cycle index
                train_mapping_idx = v[0]
                json_fold_dataset_idx[k][0] = [str(i) for i in train_mapping_idx]

                val_mapping_idx = v[1]
                json_fold_dataset_idx[k][1] = [str(i) for i in val_mapping_idx]

                # train video path
                train_video_idx = v[2]
                json_fold_dataset_idx[k][2] = str(train_video_idx)

                # val video path
                val_dataset_idx = v[3]
                json_fold_dataset_idx[k][3] = str(val_dataset_idx)

            with open((self.gait_seg_idx_path / str(self.class_num) / self.sampler / "index.json"), "w") as f:
                json.dump(json_fold_dataset_idx, f, sort_keys=True, indent=4)
                
        else:
            raise ValueError(
                "the gait seg idx path is not exist, please check the path."
            )

        return fold_dataset_idx
