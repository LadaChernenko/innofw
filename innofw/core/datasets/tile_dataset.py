# это новый датасет для prediction 
#
from pathlib import Path
#
import torch
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import image_to_tensor, to_numpy
#
from innofw.core.datasets.utils import prep_data, read_bands_from_folder
from innofw.core.constants import SegDataKeys
import logging


class TileDataset(Dataset):  # todo: consider the device on which tensors are located
    def init(self, raster_band_folder, 
                    channels, 
                    ext, 
                    pred_save_path, 
                    threshold, 
                    label_folder=None, 
                    label_names=None,
                    label_ext=None,
                    tile_size=(448, 448),
                    tile_step=(448, 448), 
                    transforms=None, 
                    from_logits=True):

        self.raster_band_folder = raster_band_folder
        self.label_folder = label_folder

        self.channels = channels
        self.ext = ext
        self.label_names = label_names
        self.label_ext = label_ext

        self.transforms = transforms

        self.tile_size = tile_size
        self.tile_step = tile_step
        self.threshold = threshold  # todo:

        self.from_logits = from_logits
        self.pred_save_path = Path(pred_save_path)
        self.is_setup = False
        logging.error("finished creating tiledataset")

    def getitem(self, item):
        if not self.is_setup:
            self.setup()

        tile = self.tiles[item]
        coords = self.crops[item]
        label_tile = None if self.label_folder is None else self.label_tiles[item]

        output = prep_data(tile, label_tile, self.transforms)
        output[SegDataKeys.coords] = coords
        return output

    def len(self):
        if not self.is_setup:
            self.setup()

        return len(self.tiles)

    def setup(self):
        logging.error("setting up the tile dataset")
        large_img = read_bands_from_folder(self.raster_band_folder, self.channels, self.ext)  # read the raster

        self.tiler = ImageSlicer(large_img.shape, tile_size=self.tile_size, tile_step=self.tile_step)
        self.crops = self.tiler.crops
        self.tiles = [tile.astype(np.int32) for tile in self.tiler.split(large_img)]
        if self.label_folder is not None:
            label_img = read_bands_from_folder(self.label_folder, self.label_names, self.label_ext)
            self.label_tiler = ImageSlicer(label_img.shape, tile_size=self.tile_size, tile_step=self.tile_step)
            self.label_tiles = [tile.astype(np.int8) for tile in
                                self.label_tiler.split(label_img)]
        # np.moveaxis(, -1, 0)
        self.merger = TileMerger(self.tiler.target_shape, 1, self.tiler.weight)
        self.is_setup = True

    def add_prediction(self, pred, coords, idx):
        self.merger.integrate_batch(pred, coords)
        # logging.error(idx, self.len())
        if idx == self.len() - 1:
            self.save_predictions()

    def save_predictions(self):
        # save the prediction
        merged = self.merger.merge()
        if self.from_logits:
            merged = torch.sigmoid(merged)

        merged_mask = np.moveaxis(to_numpy(merged > self.threshold), 0, -1).astype(np.uint8)
        merged_mask = self.tiler.crop_to_orignal_size(merged_mask).squeeze()

        # todo: this might produce errors when multichannel predictions are used
        h, w = merged_mask.shape[0], merged_mask.shape[1]

        profile = {
            'count': 1,  # update this
            'dtype': np.uint8,
            'height': h,
            'width': w,
        }
        # logging.info(Path('.').resolve())  # todo fix this
        file_path = Path(self.pred_save_path,
                         f'{self.raster_band_folder.name}/pred.tif')
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with rio.open(file_path, 'w', **profile) as dst:
            dst.write(merged_mask, 1)

        self.is_setup = False