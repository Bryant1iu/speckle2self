import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from utils.image_ops import resize_image, linear_normalization  


class BaseDenoisingDataset(Dataset):
    """
    Base class for denoising datasets with shared utilities.
    """
    def __init__(self, interp='linear'):
        self.interp = interp
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.3),
            # A.ElasticTransform(alpha=1, sigma=50, interpolation=1, border_mode=4, p=0.4)
        ], additional_targets={
            'image0': 'image',
            'mask': 'image'
        })

    def preprocess_image(self, image):
        """
        Resize and normalize image at different scales.
        """
        image_low = resize_image(image, 0.25, interpol=self.interp)
        image_high = image

        image_low = linear_normalization(image_low)
        image_high = linear_normalization(image_high)

        return image_low, image_high

    def to_tensor(self, *images):
        """
        Convert numpy images (H, W) to PyTorch tensors (1, H, W).
        """
        return [torch.from_numpy(np.expand_dims(img, 0)) for img in images]


class DenoisingDatasetPaired(BaseDenoisingDataset):
    """
    Dataset for paired high-res, low-res and label images stored in separate folders.
    Expected directory structure:
        image_dir/
            img_hr/   - high resolution images
            img_lr/   - low resolution images
            label/    - clean label images
    Files are matched by sorted filename across the three folders.
    """
    def __init__(self, image_dir, interp='linear'):
        super().__init__(interp)
        hr_dir = os.path.join(image_dir, 'img_hr')
        lr_dir = os.path.join(image_dir, 'img_lr')
        label_dir = os.path.join(image_dir, 'label')

        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
                                if self._is_image(f)])
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)
                                if self._is_image(f)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                   if self._is_image(f)])

        assert len(self.hr_paths) == len(self.lr_paths) == len(self.label_paths), \
            f"Mismatched file counts: hr={len(self.hr_paths)}, lr={len(self.lr_paths)}, label={len(self.label_paths)}"

    @staticmethod
    def _is_image(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy'))

    def _load_image(self, path):
        if path.endswith('.npy'):
            return np.load(path).astype(np.float32)
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float32)

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        image_high = linear_normalization(self._load_image(self.hr_paths[idx]))
        image_low = linear_normalization(self._load_image(self.lr_paths[idx]))
        image_clean = linear_normalization(self._load_image(self.label_paths[idx]))

        transformed = self.transform(
            image=image_low,
            image0=image_high,
            mask=image_clean
        )

        image_low, image_high, image_clean = self.to_tensor(
            transformed['image'],
            transformed['image0'],
            transformed['mask']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
            'image_clean': image_clean
        }


class DenoisingDatasetCCA(BaseDenoisingDataset):
    """
    Dataset for CCA-based denoising using unsupervised low/mid/high resolution inputs.
    Loads a single numpy array of shape (N, H, W).
    """
    def __init__(self, image_dir, interp='linear'):
        super().__init__(interp)
        self.images = np.load(os.path.join(image_dir, "train_data.npy"))  # shape (N, H, W)
        self.num_images = self.images.shape[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_raw = self.images[idx]
        image_low, image_high = self.preprocess_image(image_raw)

        transformed = self.transform(
            image=image_low, image0=image_high
        )

        image_low, image_high = self.to_tensor(
            transformed['image'], transformed['image0']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
        }


class DenoisingDatasetSimulator(BaseDenoisingDataset):
    """
    Dataset for simulated denoising with paired noisy and clean images.
    Loads a single numpy array of shape (N, 2, H, W).
    """
    def __init__(self, path, interp='linear'):
        super().__init__(interp)
        data = np.load(os.path.join(path, "train_data.npy"))  # shape (N, 2, H, W)
        self.noisy_imgs = data[:, 0]
        self.clean_imgs = data[:, 1]
        self.num_images = self.noisy_imgs.shape[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        noisy = self.noisy_imgs[idx]
        clean = linear_normalization(self.clean_imgs[idx])

        image_low, image_high = self.preprocess_image(noisy)

        transformed = self.transform(
            image=image_low,
            image0=image_high,
            mask=clean
        )

        image_low, image_high, image_clean = self.to_tensor(
            transformed['image'],
            transformed['image0'],
            transformed['mask']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
            'image_clean': image_clean
        }