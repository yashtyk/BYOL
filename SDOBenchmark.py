import datetime as dt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, List, Union, Sequence

from PIL import Image
import pandas as pd
import numpy as np
import copy

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


class BaseDataset (Dataset, ABC):
    @abstractmethod
    def y(self, indices: Optional[Sequence[int]] = None) -> List:
        pass



class SDOBenchmarkDataset_time_steps(BaseDataset):
    def __init__(
        self,
        csv_file: Path,
        root_folder: Path,
        channel="171",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        time_steps: Union[int, List[int]] = 0,
        non_flare_only = True,
        ls = None,
        is_train = True
    ):
        metadata = pd.read_csv(csv_file, parse_dates=["start", "end"])
        self.is_train = is_train
        self.non_flare_only = non_flare_only
        self.treshold = 1e-6
        self.root_folder = root_folder
        self.channel = channel
        self.transform = transform
        self.target_transform = target_transform

        self.time_steps_values = [0, 7 * 60, 10 * 60 + 30, 11 * 60 + 50]
        self.time_steps = time_steps if isinstance(time_steps, list) else [time_steps]


        if ls is None:
            self.setup(metadata)
        else:
            self.ls = ls

    def setup(self, metadata):
        ls = []
        to_tensor  = transforms.ToTensor()
        for i in range(len(metadata)):
            is_step=[]
            sample_metadata = metadata.iloc[i]
            target = sample_metadata["peak_flux"]
            if self.non_flare_only and target > self.treshold:
                continue
            if self.target_transform is not None and \
                isinstance(self.target_transform(target), int) and \
                self.target_transform(target) < 0:
                # Ignore sample if it is not part of a class
                continue

            sample_active_region, sample_date = sample_metadata["id"].split("_", maxsplit=1)

            paths: List[Path] = []
            for time_step in self.time_steps:
                image_date = sample_metadata["start"] + dt.timedelta(minutes=self.time_steps_values[time_step])
                image_date_str = dt.datetime.strftime(image_date, "%Y-%m-%dT%H%M%S")
                image_name = f"{image_date_str}__{self.channel}.jpg"
                paths.append(Path(sample_active_region) / sample_date / image_name)



            if not all((self.root_folder / path).exists() for path in paths):
                paths1 = [path for path in paths if (self.root_folder / path).exists()]
                is_step = [idx for idx, path in enumerate(paths) if not (self.root_folder / path).exists()]
                if len(paths1) < 3:
                    continue
                else:
                    paths = paths1

            # check and delete black images
            for idx, path in enumerate(paths):
                image = Image.open(self.root_folder / path)
                image = to_tensor(image)
                if image.std() == 0:
                    paths.remove(path)
                    is_step.append(idx)

            # check if 3 time steps exist

            if len(paths) < 3:
                continue

            if len(paths) == 3:
                if is_step[0]<3:
                    num_replace = is_step[0]
                else:
                    num_replace = 2

                img_replace = copy.deepcopy(paths[num_replace])

                paths.insert(is_step[0], img_replace)





            ls.append((paths, target))

        self.ls = ls
    def transform_byol(self, x):
        p = random.random()
        if p <= 0.3:

            brightness = 0.8
            hue = 0.2
            chosen_brightness = random.uniform(max(0, 1-brightness), 1+brightness)
            chosen_hue = random.uniform(-hue, hue)

            x_t = [F.adjust_brightness(x[i], chosen_brightness) for i in range ( 4)]
            x = x_t
            x_t = [F.adjust_hue(x[i], chosen_hue) for i in range (4)]
            x= x_t

        p = random.random()

        if p <= 0.5:
            x_t = [F.hflip(x[i]) for i in range( 4)]
            x= x_t

        p = random.random()

        if p <= 0.2:
            kernel_size =(3, 3)
            sigma= random.uniform(1.0, 2.0)
            x_t = [F.gaussian_blur(x[i], kernel_size, [sigma, sigma]) for i in range (4)]
            x= x_t
        #random resized crop
        size = 256
        i, j, h, w = transforms.RandomResizedCrop.get_params(x[0], (0.8, 1), (1, 1))
        x_t = [F.resized_crop(x[m], i, j, h, w, size=(size, size) ) for m in range(4)]
        x= x_t

        return x









    def __len__(self) -> int:
        return len(self.ls)

    def __getitem__(self, index):
        metadata = self.ls[index]
        target = metadata[1]
        images = [Image.open(self.root_folder / path) for path in metadata[0]]
        to_tensor = transforms.ToTensor()
        images = [to_tensor(image) for image in images]
        if self.is_train:
            image1 = self.transform_byol(images)
            image2 = self.transform_byol(images)

        if self.transform:
            images = [self.transform(image) for image in images]
            image1 = [self.transform(image) for image in image1]
            image2 = [self.transform(image) for image in image2]

        if self.target_transform:
            target1 = self.target_transform(target)

        if not torch.is_tensor(images[0]):
            return images[0], target

        # Put images of different time steps as one image of multiple channels (time steps ~ rgb)
        image = torch.cat(images, 0)
        images1 = torch.cat(image1, 0)
        images2 = torch.cat(image2, 0)

        return image, images1, images2, target1

    def y(self, indices: Optional[Sequence[int]] = None) -> list:
        ls = self.ls
        if indices is not None:
            ls = (self.ls[i] for i in indices)

        #if self.target_transform is not None:
        #    return [self.target_transform(y[1]) for y in ls]

        return [y[1] for y in ls]


class SDOBenchmarkDataset_time_step_pairs(SDOBenchmarkDataset_time_steps):
    def __init__(
            self,
            csv_file: Path,
            root_folder: Path,
            channel="171",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            time_steps: Union[int, List[int]] = 0,
            indx: List[int] = [0],
            non_flare_only = False,
            ls= None

    ):
        super().__init__( csv_file, root_folder, channel,transform,target_transform,time_steps, ls =ls)
        self.indx = indx
        self.non_flare_only = non_flare_only
        self.make_pairs()
    def make_pairs(self):
        idx = self.indx.copy()
        np.random.shuffle(idx)
        pairs = [[idx[2*i], idx[2*i+1]] for i in range(0, len(idx)//2)]
        self.pairs = pairs

        print(pairs[0])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index):
        ind1 = self.pairs[index][0]
        ind2 = self.pairs[index][1]
        img1, target, _ =SDOBenchmarkDataset_time_steps.__getitem__(self, ind1)
        img2, _, _ = SDOBenchmarkDataset_time_steps.__getitem__(self, ind2)

        return img1, img2, target






