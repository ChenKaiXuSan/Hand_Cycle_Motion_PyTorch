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
Last Modified: Friday June 14th 2024 1:21:22 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import os, json, shutil, copy, random, re
from typing import Any, Dict, List, Tuple

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import KFold
from pathlib import Path

class DefineCrossValidation(object):
    """process:
    cross validation > over/under sampler > train/val split
    fold: [train/val]: [path]
    """

    def __init__(self, config) -> None:
        
        self.video_path: Path = Path(config.data.data_path)
        self.index_path: Path = Path(config.data.index_path)
        self.root_path: Path = self.video_path.parent.parent

        self.K: int = config.train.fold
        # self.sampler: str = config.data.sampling # data balance, [over, under, none]
        self.val_ratio: float = config.data.val_ratio

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

    def make_dataset_with_video(self, dataset_idx: list, fold: int, flag: str, X: list, y: list):

        target_path = self.video_path.parent.parent / f"fold{fold}" / 'data' / flag
        target_index_path = self.index_path.parent.parent / f"fold{fold}" / 'index_mapping' / flag

        for i in dataset_idx:
            video_path = X[i] 
            video_label = y[i]
            class_type = str(video_path).split("/")[-4]
            shape = str(video_path).split("/")[-3]

            index_path = '/'.join(str(video_path).split('/')[:-5]) + '/index_mapping/' + str(video_path).split('/')[6] + f'/{shape}/' + str(video_path).split('/')[-1].replace('mp4', 'json')

            _t_path = target_path / class_type 
            _t_index_path = target_index_path 
            
            if not _t_path.exists():
                _t_path.mkdir(parents=True)

            if not _t_index_path.exists():
                _t_index_path.mkdir(parents=True)

            shutil.copy(video_path, _t_path / video_path.name)
            shutil.copy(index_path, _t_index_path / Path(video_path.name.split('.')[0] +".json"))

        print(f"fold {fold} {flag} dataset has been created.")        

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
    
    def get_total_dataset(self, class_num: int, raw_video_path: Path) -> Dict:

        res_dict = {}
        for one_class in raw_video_path.iterdir():
            res_dict[one_class.name] = []
        
        # make inverse dict
        # TODO:　这里有问题，因为更改了目录树
        str_list = res_dict.keys()
        # def extract_numbers(s):
        #     left, right = re.findall(r'\d+', s)
        #     return int(left), int(right)
        
        # inverse_dict = sorted(str_list, key=extract_numbers)
        inverse_dict = sorted(str_list)
        inverse_dict = {v: k for k, v in enumerate(inverse_dict)}

        # save inverse dict 
        with open(self.root_path / 'class_to_num.json', 'w') as f:
            json.dump(inverse_dict, f, indent=4)

        final_dict = {}

        X, y = [], []

        for one_class in raw_video_path.iterdir():
            
            path = list(one_class.glob('**/*.mp4'))
            label = [inverse_dict[one_class.name]] * len(path)

            X.extend(path)
            y.extend(label)
            

        return X, y

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

        X, y = self.get_total_dataset(self.class_num, self.video_path)
        
        # define the cross validation
        # X: video path, in path.Path foramt. len = 1954
        # y: label, in list format. len = 1954, type defined by class_num_mapping_Dict.
        # groups: different patient, in list format. It means unique patient index. [54]
        # X, y, groups = self.process_cross_validation(total_dataset)

        # sgkf = StratifiedGroupKFold(n_splits=K)
        kfold = KFold(n_splits=K, shuffle=True, random_state=42)

        circle_kfold = list(kfold.split(X=X, y=y))

        for f in range(K):
            self.make_dataset_with_video(circle_kfold[f][0], f, "train", X, y)
            self.make_dataset_with_video(circle_kfold[f][1], f, "val", X, y)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        flag = False
        for i in range(self.K):
            t_path = self.root_path / f"fold{i}"
            if t_path.exists(): continue;
            else: 
                flag = True
                break;
        
        if flag:
        
            # * step 1: get total data
            self.prepare()

        # * step 2: get the data path    
        res_dict = {}

        for k in range(self.K):
            res_dict[k] = {
                "train": self.root_path / f"fold{k}" / 'data' / "train",
                "val": self.root_path / f"fold{k}" / 'data' / "val",
                "train_index": self.root_path / f"fold{k}" / 'index_mapping' / "train",
                "val_index": self.root_path / f"fold{k}" / 'index_mapping' / "val",
            }
        
        return res_dict