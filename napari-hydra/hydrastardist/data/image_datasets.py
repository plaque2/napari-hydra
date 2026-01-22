from abc import ABC
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
from albumentations.core.serialization import Serializable as Transform
from tensorflow.keras.utils import Sequence, load_img, to_categorical


class ImageAbstractDataset(Sequence, ABC):

    def __init__(
        self,
        img_dir: Union[str, Path],
        transform: Transform = None,
        batch_size: int = 50,
        seed: Union[None, int, np.random.Generator] = None,
        **kwargs,
    ) -> None:
        """ Initializes the dataset.

        Args:
            img_dir (Union[str, Path]): Directory containing images to use.
            transform (Transform, optional): Albumentations transforms to apply to images.
            batch_size (int, optional): Size of each batch. Defaults to 50.
            epoch_size (int, optional): Total number of samples in each epoch. Defaults to 1000.
        """
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(seed)

        self._img_list = list(sorted(Path(img_dir).rglob("*.tif")))

        self._transform = transform

        self.img_names = [img.name for img in self._img_list]
        self.batch_size = batch_size

    def _generate_data(self, batch_indices: Iterable[int]) -> np.ndarray:
        """ Given indices of images, generates a batch of samples from them

        Args:
            batch_indices (Iterable[int]): Indices of images to be included in the batch

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of arrays representing images and masks
        """
        images = []
        for img_idx in batch_indices:
            img = load_img(self._img_list[img_idx], color_mode="grayscale")
            img = np.expand_dims(img, -1)
            if self._transform is not None:
                transformed = self._transform(image=img)
                img = transformed["image"]
            images.append(img)

        return np.stack(images).astype(np.float32)

class ImageRandomDataset(ImageAbstractDataset):
    """ A dataset containing microscopy images.

        Returns a random subset of samples in each batch.
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        transform: Optional[Transform] = None,
        batch_size: int = 50,
        epoch_size: int = 1000,
        seed: Union[None, int, np.random.Generator] = None,
        **kwargs,
    ) -> None:
        """ Initializes the dataset.

        Args:
            img_dir (Union[str, Path]): Directory containing images to use.
            transform (Transform, optional): Albumentations transforms to apply to images and masks.
            batch_size (int, optional): Size of each batch. Defaults to 50.
            epoch_size (int, optional): Total number of samples in each epoch. Defaults to 1000.
        """
        super().__init__(img_dir, transform, batch_size, seed, **kwargs)

        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return int(np.ceil(self.epoch_size / self.batch_size))

    def __getitem__(self, index: int) -> np.ndarray:
        """ Generates a random batch of images.

            Index argument is not used, it is only an argument because Sequence requires it.
        """
        batch_indices = self._rng.choice(len(self._img_list), size=self.batch_size, replace=True)
        images = self._generate_data(batch_indices)

        return images