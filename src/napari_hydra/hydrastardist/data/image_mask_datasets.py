from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
from albumentations.core.serialization import Serializable as Transform
from tensorflow.keras.utils import Sequence as KerasSequence
from tensorflow.keras.utils import load_img, to_categorical


def binarize_mask(mask: np.ndarray, max_value: int = 255) -> np.ndarray:
    mask = mask / max_value
    return np.rint(mask).astype(int)

class ImageMaskAbstractDataset(KerasSequence, ABC):

    def __init__(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        num_classes: int = 1,
        transform: Optional[Transform] = None,
        batch_size: int = 50,
        seed: Union[None, int, np.random.Generator] = None,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        """ Initializes the dataset.

        Args:
            img_dir (Union[str, Path]): Directory containing images to use.
            mask_dir (Union[str, Path]): Directory containing masks corresponding to the images.
            transform (Transform, optional): Albumentations transforms to apply to images and masks.
            batch_size (int, optional): Size of each batch. Defaults to 50.
            epoch_size (int, optional): Total number of samples in each epoch. Defaults to 1000.
            num_samples (int, optional): Number of samples to subsample from the dataset.

        Raises:
            ValueError: If any image does not match to the mask.
        """
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(seed)

        self._img_list = np.sort(list(Path(img_dir).rglob("*.tif")))
        self._mask_list = np.sort(list(Path(mask_dir).rglob("*.tif")))

        if len(self._img_list) != len(self._mask_list):
            raise RuntimeError("The number of images and masks does not match")

        for img_path, mask_path in zip(self._img_list, self._mask_list):
            if not mask_path.stem.startswith(img_path.stem):
                raise ValueError(
                    f"The image {img_path.name} does not match the mask {mask_path.name}."
                )

        if num_samples is not None:
            indices = self._rng.choice(
                np.arange(len(self._img_list)),
                size=num_samples,
                replace=False,
            )
            self._img_list = self._img_list[indices]
            self._mask_list = self._mask_list[indices]

        self._transform = transform

        self._num_classes = num_classes
        self._batch_size = batch_size

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def img_names(self) -> Sequence[str]:
        return [img.name for img in self._img_list]


    def _generate_data(self, batch_indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """ Given indices of images, generates a batch of samples from them

        Args:
            batch_indices (Iterable[int]): Indices of images to be included in the batch

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of arrays representing images and masks
        """
        images, masks = [], []
        for img_idx in batch_indices:
            img = load_img(self._img_list[img_idx], color_mode="grayscale")
            mask = load_img(self._mask_list[img_idx], color_mode="grayscale")
            img = np.expand_dims(img, -1)
            mask = np.array(mask)
            if self._transform is not None:
                transformed = self._transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]
            images.append(img)
            masks.append(mask)

        images_array = np.stack(images).astype(np.float32)
        masks_array = np.stack(masks).astype(np.float32)
        # rescale images to (-1, 1)
        images_array = images_array*2 - 1
        return images_array, masks_array

    def _expand_masks(self, masks):
        if self.num_classes == 1:
            masks = np.expand_dims(masks, -1)
            masks = binarize_mask(masks)
        else:
            masks = to_categorical(masks, num_classes=self.num_classes)

        return masks


class ImageMaskRandomDataset(ImageMaskAbstractDataset):
    """ A dataset containing microscopy images and corresponding masks.

        Returns a random subset of samples in each batch.
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        num_classes: int = 1,
        transform: Transform = None,
        batch_size: int = 50,
        epoch_size: int = 1000,
        max_iters: int = 10,
        threshold_obj_perc: float = 0.01,
        seed: Union[None, int, np.random.Generator] = None,
        expand_masks: bool = True,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        """ Initializes the dataset.

        Args:
            img_dir (Union[str, Path]): Directory containing images to use.
            mask_dir (Union[str, Path]): Directory containing masks corresponding to the images.
            transform (Transform, optional): Albumentations transforms to apply to images and masks.
            batch_size (int, optional): Size of each batch. Defaults to 50.
            epoch_size (int, optional): Total number of samples in each epoch. Defaults to 1000.
            max_iters (int, optional): Max number of resampling if batches don"t satisfy condition. Defaults to 10.
            threshold_obj_perc (float, optional): Required ROI density for the batch. Defaults to 0.01.
            num_samples (int, optional): Number of samples to subsample from the dataset.

        Raises:
            ValueError: If any image does not match to the mask.
        """
        super().__init__(img_dir, mask_dir, num_classes, transform, batch_size, seed, num_samples, **kwargs)

        self._max_iters = max_iters
        self._threshold_obj_perc = threshold_obj_perc
        self.expand_masks = expand_masks

        self.epoch_size = epoch_size

    def __len__(self):
        return int(np.ceil(self.epoch_size / self.batch_size))

    def _check_threshhold_condition(self, masks: np.ndarray) -> bool:
        """ Given a batch of masks, checks if they satisfy the condition of ROI density
        """
        num_positive = np.sum(masks.astype(bool))
        num_pixels = np.prod(masks.shape)

        return (num_positive / num_pixels) > self._threshold_obj_perc

    def __getitem__(self, index: int):
        """ Generates a random batch of images.

            Index argument is not used, it is only an argument because Sequence requires it.
        """
        for _ in range(self._max_iters):
            batch_indices = self._rng.choice(len(self._img_list), size=self.batch_size, replace=True)
            images, masks = self._generate_data(batch_indices)
            if self._check_threshhold_condition(masks):
                break

        if self.expand_masks:
            masks = self._expand_masks(masks)

        return images.astype(np.float32), masks.astype(np.float32)


class ImageMaskDataset(ImageMaskAbstractDataset):
    """ A dataset containing microscopy images and corresponding masks.

        Returns each image only once.
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        num_classes: int = 1,
        transform: Transform = None,
        batch_size: int = 50,
        shuffle: bool = False,
        seed: Union[None, int, np.random.Generator] = None,
        expand_masks: bool = True,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        """ Initializes the dataset.

        Args:
            img_dir (Union[str, Path]): Directory containing images to use.
            mask_dir (Union[str, Path]): Directory containing masks corresponding to the images.
            transform (Transform, optional): Albumentations transforms to apply to images and masks.
            batch_size (int, optional): Size of each batch. Defaults to 50.
            shuffle (bool, optional): Whether to reshuffle the dataset after each epoch. Defaults to False.
        Raises:
            ValueError: If any image does not match to the mask.
        """
        super().__init__(img_dir, mask_dir, num_classes, transform, batch_size, seed, num_samples, **kwargs)

        self._shuffle = shuffle
        self.expand_masks = expand_masks

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self._img_list) / self.batch_size))

    def on_epoch_end(self):
        num_images = len(self._img_list)
        if self._shuffle:
            self.indices = self._rng.permutation(num_images)
        else:
            self.indices = np.arange(num_images)

    def __getitem__(self, index: int):
        """ Returns a batch of images.
        """
        batch_indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]
        images, masks = self._generate_data(batch_indices)

        if self.expand_masks:
            masks = self._expand_masks(masks)

        return images.astype(np.float32), masks.astype(np.float32)


class ImageMaskStaticDataset(KerasSequence):
    """ Dataset that always returns the same patches, can be used for plotting
        the outputs of diffrent models.
    """

    def __init__(
        self,
        npz_file_dir,
        ) -> None:
        super().__init__()
        self.npz_file_dir = Path(npz_file_dir)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(self.npz_file_dir / f"batch_{index}.npz")
        return data["images"], data["masks"]

    def __len__(self) -> int:
        return len(list(self.npz_file_dir.iterdir()))

    @staticmethod
    def create_from_directory(
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        npz_file_dir: Union[str, Path],
        num_classes: int = 1,
        transform: Transform = None,
        batch_size: int = 50,
        epoch_size: int = 1000,
        max_iters: int = 10,
        threshold_obj_perc: float = 0.01,
        seed: Union[None, int, np.random.Generator] = None,
        expand_masks: bool = True,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> ImageMaskStaticDataset:

        npz_file_dir = Path(npz_file_dir)
        random_dataset = ImageMaskRandomDataset(
            img_dir,
            mask_dir,
            num_classes,
            transform,
            batch_size,
            epoch_size,
            max_iters,
            threshold_obj_perc,
            seed,
            expand_masks,
            num_samples,
            **kwargs,
        )

        for batch, (imgs, masks) in enumerate(random_dataset):
            np.savez(npz_file_dir / f"batch_{batch}.npz", images=imgs, masks=masks)

        return ImageMaskStaticDataset(npz_file_dir)
