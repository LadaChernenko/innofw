import sys

# sys.path.append("/home/qazybek/repos/qb_utils/")

from pathlib import Path
from typing import List
from typing import Iterable, Any
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import logging
import random

from omegaconf import DictConfig
import hydra
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import image_to_tensor, to_numpy
import numpy as np
import rasterio as rio
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_toolbelt.inference import tta
import albumentations as A
from pydantic import validate_arguments
from torchmetrics import JaccardIndex, F1Score
import pandas as pd
from pytorch_lightning import Trainer
# from segmentation.models.segmentation import SeparatedBoundariesModel
# from operations.vectorize import raster_array_to_shp
# from segmentation.models.segmentation import SegmentationLM
from innofw.core.models.torch.lightning_modules import SemanticSegmentationLightningModule

from innofw.utils.framework import (
    get_callbacks,
    get_optimizer,
    get_ckpt_path,
    get_datamodule,
    get_losses,
    get_model,
    get_obj,
    map_model_to_framework,
)
from innofw.utils.getters import get_trainer_cfg
from innofw.constants import Stages

import warnings
warnings.filterwarnings("ignore")


def parse_dir(folder, channels, extensions):
    return [
        list(folder.rglob(f"*{ch}{ext}"))[0] for ext in extensions for ch in channels
    ]


def read_band_file(file_path) -> np.array:
    return rio.open(file_path).read(1)


def read_bands_from_folder(folder, channels, extensions: List) -> np.array:
    files = parse_dir(folder, channels, extensions)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(read_band_file, files)

    combined = np.dstack(results)
    return combined


def tensor2array(tensor):
    return tensor.detach().cpu().numpy()


@validate_arguments
def infer(
    data_path: Path,
    dst_path: Path,
    model,
    device,
    CHANNELS: List[str],
    tile_size: int,
    tile_step: int,
    with_padding: bool,
    with_tta: bool,
    threshold: float,
    save_geom: bool,
    display: bool,
    save_name: str = "prediction.tif",
    dry_run: bool = False,
):
    # set up output path
    # name = f"/mnt/datastore/GIS/qazybek/projects/roads/shapes/test/281222-roads-test/{data_path.stem}/"
    out_path = Path(
        dst_path,
        f"{data_path.stem}/{tile_size}_{tile_step}_tta_{with_tta}_pad_{with_padding}_thrsh_{threshold}",
    )
    out_path.mkdir(exist_ok=True, parents=True)
    # print(data_path)
    if dry_run:
        return out_path / save_name

    if data_path.stem == "label":
        return

    model = model.to(device)

    raster = rio.open(data_path)
    # print(raster.read())
    ch_count = raster.count
    if ch_count > len(CHANNELS):
        ch_count = len(CHANNELS)
    large_img = np.dstack([raster.read(i) for i in range(1, ch_count + 1)])

    # create image slicer
    tiler1 = ImageSlicer(large_img.shape, tile_size=tile_size, tile_step=tile_step)
    # get coordinates of an each slice
    crops = tiler1.crops
    # create tile merger with channels=1(as it's a binary segmentation)
    merger_1 = TileMerger(tiler1.target_shape, 1, tiler1.weight)
    # print(merger_1)
    # list to store predictions
    results_1 = []

    with torch.no_grad():
        # get all tiles from a large raster
        tiles1 = [tile for tile in tiler1.split(large_img)]
        for j, tile1 in enumerate(tqdm(tiles1)):
            if with_padding:
                aug = A.PadIfNeeded(
                    min_height=tile_size + 128, min_width=tile_size + 128, p=1
                )
                augmented = aug(image=tile1)

                # image padded
                tile1 = augmented["image"]

            img1 = torch.tensor(tile1).to(device).unsqueeze(0)
            img1 = torch.moveaxis(img1, -1, 1).float()
            img1 = torch.vstack([img1, img1])
            if with_tta:
                result = tta.d4_image2mask(model, img1)
            else:
                result = model(img1)

            i = 0
            # apply sigmoid to the prediction, then convert to array and apply thresholding
            # for both model predictions
            output = tensor2array(torch.sigmoid(result))[i][0] > threshold
            # store the result
            results_1.append(output)

    if with_padding:
        results_1 = [i[64:-64, 64:-64] for i in results_1]

    # add all predictions to the merger
    merger_1.integrate_batch(torch.tensor(results_1), crops)

    # merge slices
    merged_1 = merger_1.merge()

    # convert to numpy and move channels dim
    merged_1 = np.moveaxis(merged_1.detach().cpu().numpy(), 0, -1)
    # crop the mask(as before division the padding is applied)
    final_pred_1 = tiler1.crop_to_orignal_size(merged_1).squeeze()
    # get the metadata
    meta = rio.open(data_path).meta  # jp2  # / "RED.tif"
    meta["count"] = 1
    meta["driver"] = "GTiff"

    # save the prediction raster
    with rio.open(out_path / save_name, "w", **meta) as f:
        f.write(final_pred_1, 1)

    if save_geom:
        raster_array_to_shp(out_path / save_name, data_path, out_path)

    if display:
        plt.figure(figsize=(20, 20))
        plt.imshow(rio.open(out_path / save_name).read(1))

    return out_path / save_name


def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))



def sort_result(df):
    print('Params for the best iou:')
    print(df.sort_values('iou', ascending=False).head(2))

    print('Params for the best f1 score:')
    print(df.sort_values('f1', ascending=False).head(2))

@hydra.main(config_path="config/", config_name="experiment_landslides.yaml", version_base="1.2")
def main(cfg):
    
    # model = hydra.utils.instantiate(cfg.model)
    trainer_cfg = get_trainer_cfg(cfg)
    trainer = Trainer()
    task = cfg.get("task")
    model = get_model(cfg.models, trainer_cfg)

    trainer = Trainer(trainer_cfg)
    
    model_lm = SemanticSegmentationLightningModule.load_from_checkpoint(
        cfg.ckpt_path, model=model, optimizer_cfg=cfg.optimizers, losses=cfg.losses, scheduler_cfg=None
    )
    # print(model_lm)
    # # model_lm = SegmentationLM
    CHANNELS = cfg.channels
    EXT = cfg.extension 

    data_stage = Stages.predict

    device = torch.device("cuda:0")

    safe_path = cfg.datamodules.dst_path
    # print(safe_path)


    iou_score = JaccardIndex(task="binary", num_classes=1)
    f1_score = F1Score(task="binary", num_classes=1)


    # grid search the best parameters
    parameters = {
        "with_tta": [True, False],
        "with_padding": [True, False],
        "tile_size": [1024, 1536, 2048],
        "tile_step": [1024, 1536, 2048],
        "threshold": np.arange(0.2, 1, 0.2),
        "save_geom": [False],
        "display": [False],
    }
    cfg_datamodule = cfg.get("datasets")
    print(cfg_datamodule)

    framework = map_model_to_framework(model)
    print(framework)
    
    for settings in grid_parameters(parameters):
        # cfg_datamodule["with_tta"] = settings["with_tta"]
        # cfg_datamodule["with_padding"] = settings["with_padding"]
        print(settings["threshold"])
        cfg_datamodule["tile_size"] = settings["tile_size"]
        cfg_datamodule["tile_step"] = settings["tile_step"]
        # cfg_datamodule["threshold"] = settings["threshold"]

        
        datamodule = get_datamodule(cfg_datamodule, 
                                    framework, 
                                    task=task, 
                                    # batch_size=cfg.get("batch_size")
                                    )
        trainer.predict(model, datamodule)
        datamodule.save_preds(save_path)

    # framework = map_model_to_framework(model)
    # augmentations = get_obj(cfg, "augmentations", task, framework)
    # datamodule = get_datamodule(
    #     cfg.datasets,
    #     framework,
    #     task=task,
    #     stage=data_stage,
    #     augmentations=augmentations,
    #     batch_size=cfg.get("batch_size"),
    # )
    # data_path = cfg.datamodules.img_path
    # folders = list(Path(data_path).iterdir())
    # dst_path = Path(cfg.datamodules.dst_path)
    # dst_path.mkdir(exist_ok=True, parents=True)
    # df = pd.DataFrame()

    # callbacks = get_callbacks(
    #     cfg, task, framework, metrics=metrics, losses=losses, datamodule=datamodule
    # )
    # # create logger with 'spam_application'
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # # create file handler which logs even debug messages
    # fh = logging.FileHandler("inference.log")
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    """qb notes:
        код должен будет вызываться следующим образом:

        trainer = Trainer(...)
        model = Model(...)

        for settings in hparams:
            # обновляешь параметры датамодуля
            # надо будет добавить доп. параметры в датамодуль
            cfg_datamodule = cfg.get("datasets")
            cfg_datamodule["tile_size"] = settings['tile_size"]
            # ...

            datamodule = get_datamodule(cfg_datamodule)
            trainer.predict(model, datamodule)  # вот здесь тренер использует у датамодуля predict_dataloader(), 
            а у модели вызывает model.predic_step()
            # получается ты эти две функции реализуешь. (код отправил в тг.)
            datamodule.save_preds(save_path)  # generate folder_name from settings

            # compute metrics
            compute_metrics(save_path, datamodule.predict_dataset.masks)

            # если видишь решение лучше, то сделай так как видишь
    """


    # # for data_path, mask_path in zip(files, mask_files):
    # for folder in tqdm(random.sample(folders, 10)):
    #     for settings in grid_parameters(parameters):
    #         try:
    #             logger.debug(f"{folder.name} - {settings}")
    #             data_path = folder / f"{folder.name}.tif"
    #             mask_path = folder / f"{cfg.label}.tif"
    #             pred_path = infer(
    #                 data_path,
    #                 dst_path,
    #                 model_lm,
    #                 device,
    #                 CHANNELS,
    #                 **settings,
    #                 dry_run=True,
    #             )
    #             pred, mask = rio.open(pred_path).read(1), rio.open(mask_path).read(1)
    #             pred, mask = [torch.tensor(i) for i in [pred, mask]]
    #             score = {"iou": iou_score(pred, mask), "f1": f1_score(pred, mask)}
    #             logger.debug(f"{folder.name}: iou-{score['iou']}, f1-{score['f1']}")
    #             settings["name"] = folder.name
    #             df = pd.concat([df, pd.Series(settings | score).to_frame().T], ignore_index=1)
    #         except:  #  ValueError
    #             pass
    #         result_path = Path(dst_path, "results.csv")
    #         df.to_csv(result_path, index=False)
    #         sort_result(df)
    #         print('Results csv are saved at', result_path)


if __name__ == '__main__':
    main()
