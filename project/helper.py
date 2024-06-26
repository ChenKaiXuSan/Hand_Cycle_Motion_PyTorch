import os, logging, json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    FullGrad,
    GradCAMPlusPlus,
    AblationCAM,
    ScoreCAM,
    LayerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import visualization as viz


def save_helper(config, model, dataloader, fold):

    if config.train.experiment == "late_fusion":
        total_pred, total_label = save_inference_late_fusion(
            config, model, dataloader, fold
        )

    else:
        total_pred, total_label = save_inference(config, model, dataloader, fold)

    save_metrics(total_pred, total_label, fold, config)
    save_CM(total_pred, total_label, fold, config)


def save_inference_late_fusion(config, model, dataloader, fold):

    total_pred_list = []
    total_label_list = []

    device = f"cuda:{config.train.gpu_num}"

    test_dataloader = dataloader.test_dataloader()
    stance_cnn = model.stance_cnn
    swing_cnn = model.swing_cnn

    for i, batch in enumerate(test_dataloader):

        # if i == 5: break; # ! debug only

        stance_video = (
            batch["video"][..., 0].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, c, t, h, w
        swing_video = (
            batch["video"][..., 1].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, c, t, h, w
        # sample_info = batch["info"] # b is the video instance number

        label = batch["label"]

        stance_cnn.eval().to(device)
        swing_cnn.eval().to(device)

        with torch.no_grad():

            stance_preds = stance_cnn(stance_video)
            swing_preds = swing_cnn(swing_video)

        predict = (stance_preds + swing_preds) / 2
        preds_softmax = torch.softmax(predict, dim=1)

        # * Since saving the video tensor is too GPU memory intensive, the content is extracted in a batch to be saved
        # FIXME: numpy.core._exceptions._ArrayMemoryError
        random_index = random.sample(range(0, stance_video.size()[0]), 2)
        # save_CAM(
        #     config, stance_cnn, stance_video, label, fold, "stance", i, random_index
        # )
        # save_CAM(config, swing_cnn, swing_video, label, fold, "swing", i, random_index)

        for i in preds_softmax.tolist():
            total_pred_list.append(i)
        for i in label.tolist():
            total_label_list.append(i)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    # save the results
    save_path = Path(config.train.log_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{config.model.model}_{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{config.model.model}_{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path}/{config.model.model}_{fold}"
    )

    return pred, label


def save_inference(config, model, dataloader, fold):

    total_pred_list = []
    total_label_list = []

    test_dataloader = dataloader.test_dataloader()

    for i, batch in enumerate(test_dataloader):
        
        # if i>10: break # only for debug

        # input and label
        video = (
            batch["video"].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, c, t, h, w
        label = (
            batch["label"].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, class_num

        model.eval().to(f"cuda:{config.train.gpu_num}")

        # pred the video frames
        with torch.no_grad():
            preds = model(video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1:
            preds = preds.squeeze(dim=-1)
            preds_softmax = torch.softmax(preds, dim=1)
        else:
            preds_softmax = torch.softmax(preds, dim=1)

        # FIXME: numpy.core._exceptions._ArrayMemoryError
        # ! the batch size can not be 1.
        random_index = random.sample(range(0, video.size()[0]), 2)
        # save_CAM(
        #     config,
        #     model.video_cnn,
        #     video,
        #     label,
        #     fold,
        #     config.train.experiment,
        #     i,
        #     random_index,
        # )

        for i in preds_softmax.tolist():
            total_pred_list.append(i)
        for i in label.tolist():
            total_label_list.append(i)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    # save the results
    save_path = Path(config.train.log_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{config.model.model}_{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{config.model.model}_{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path} / {config.model.model}_{fold}"
    )

    return pred, label


def save_metrics(all_pred, all_label, fold, config):
    """save the final metrics into the log file, cross whole pred and label.

    Args:
        all_pred (torch.tensor): all the prediction of the model.
        all_label (torch.tensor): all the label of the model.
        fold (int): the fold number.
        config (hydra): the config file.
    """

    save_path = Path(config.train.log_path) / "metrics.txt"

    # define metrics
    # num_class = torch.unique(all_label).size(0)
    num_class = config.model.model_class_num
    _accuracy = MulticlassAccuracy(num_class)
    _precision = MulticlassPrecision(num_class)
    _recall = MulticlassRecall(num_class)
    _f1_score = MulticlassF1Score(num_class)
    _auroc = MulticlassAUROC(num_class)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    logging.info("*" * 100)
    logging.info("accuracy: %s" % _accuracy(all_pred, all_label))
    logging.info("precision: %s" % _precision(all_pred, all_label))
    logging.info("recall: %s" % _recall(all_pred, all_label))
    logging.info("f1_score: %s" % _f1_score(all_pred, all_label))
    logging.info("aurroc: %s" % _auroc(all_pred, all_label.long()))
    logging.info("confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))
    logging.info("#" * 100)

    with open(save_path, "a") as f:
        f.writelines(f"Fold {fold}\n")
        f.writelines(f"accuracy: {_accuracy(all_pred, all_label)}\n")
        f.writelines(f"precision: {_precision(all_pred, all_label)}\n")
        f.writelines(f"recall: {_recall(all_pred, all_label)}\n")
        f.writelines(f"f1_score: {_f1_score(all_pred, all_label)}\n")
        f.writelines(f"aurroc: {_auroc(all_pred, all_label.long())}\n")
        f.writelines(f"confusion_matrix: {_confusion_matrix(all_pred, all_label)}\n")
        f.writelines("#" * 100)
        f.writelines("\n")


def save_CM(all_pred, all_label, fold, config):
    """save the confusion matrix into file.

    Args:
        all_pred (torch.tensor): all the prediction of the model.
        all_label (torch.tensor): all the label of the model.
        fold (int): the fold number.
        config (hydra): the config file.
    """

    class_to_num_path = config.data.class_to_num
    with open(class_to_num_path, "r") as f:
        class_to_num = json.load(f)

    save_path = Path(config.train.log_path) / "CM"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    # define metrics
    # num_class = torch.unique(all_label).size(0)
    num_class = config.model.model_class_num
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    logging.info("_confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))

    # 设置字体和标题样式
    plt.rcParams.update({"font.size": 30, "font.family": "sans-serif"})

    confusion_matrix_data = _confusion_matrix(all_pred, all_label).cpu().numpy() * 100

    axis_labels = [i for i in class_to_num.keys()]

    # 使用matplotlib和seaborn绘制混淆矩阵
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        vmin=0,
        vmax=100,
    )
    plt.title(f"Fold {fold} (%)", fontsize=30)
    plt.ylabel("Actual Label", fontsize=30)
    plt.xlabel("Predicted Label", fontsize=30)

    plt.savefig(
        save_path / f"fold{fold}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )

    logging.info(
        f"save the confusion matrix into {save_path}/fold{fold}_confusion_matrix.png"
    )


def save_CAM(
    config,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    inp_label,
    fold,
    flag,
    i,
    random_index,
):

    # guided grad cam method
    target_layer = [model.blocks[-2].res_blocks[-1]]
    # target_layer = [model.model.blocks[-2]]

    cam = GradCAMPlusPlus(model, target_layer)

    # save the CAM
    save_path = Path(config.train.log_path) / "CAM" / f"fold{fold}" / flag

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    for idx, num in enumerate(random_index):

        grayscale_cam = cam(
            input_tensor[num : num + 1], aug_smooth=True, eigen_smooth=True
        )
        output = cam.outputs

        # prepare save figure
        inp_tensor = (
            input_tensor[num].permute(1, 2, 3, 0)[-1].cpu().detach().numpy()
        )  # display original image
        cam_map = grayscale_cam.squeeze().mean(axis=2, keepdims=True)

        figure, axis = viz.visualize_image_attr(
            cam_map,
            inp_tensor,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            cmap="jet",
            title=f"label: {int(inp_label[num])}, pred: {output.argmax(dim=1)[0]}",
        )

        figure.savefig(
            save_path / f"fold{fold}_{flag}_step{i}_num{idx}_{num}.png",
        )

    # save with pytorch_grad_cam
    # visualization = show_cam_on_image(inp_tensor, cam_map, use_rgb=True)

    print(f"save the CAM into {save_path}")
