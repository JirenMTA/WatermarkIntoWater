import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

class DatasetSegWater(Dataset):
    def __init__(self, set_type, ratio, img_size, shuffle, DATASET_PATH, apply_contrast=False):
        assert set_type in {'train', 'test'}
        self.DATASET_PATH = DATASET_PATH
        self.IMAGE_PATH = os.path.join(self.DATASET_PATH, "Images")
        self.MASK_PATH = os.path.join(self.DATASET_PATH, "Masks")
        self.images_name = np.array(list(os.listdir(self.IMAGE_PATH)))
        self.masks_name = np.array(list(os.listdir(self.MASK_PATH)))
        self.data_len = len(self.images_name)
        self.img_size = img_size
        self.apply_contrast = apply_contrast

        if shuffle:
            indices = np.arange(len(self.images_name))
            np.random.shuffle(indices)
            self.images_name = self.images_name[indices]
            self.masks_name = self.masks_name[indices]
        if set_type == 'train':
            self.images_name = self.images_name[:int(self.data_len * ratio)]
            self.masks_name = self.masks_name[:int(self.data_len * ratio)]
        elif set_type == 'test':
            self.images_name = self.images_name[int(self.data_len * (1 - ratio)):]
            self.masks_name = self.masks_name[int(self.data_len * (1 - ratio)):]
        self.data_len = len(self.images_name)

    def __getitem__(self, item):
        img_path = os.path.join(self.IMAGE_PATH, self.images_name[item])
        mask_path = os.path.join(self.MASK_PATH, self.masks_name[item])

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.apply_contrast:
            img = self._apply_contrast(img)

        size_pixel = 2**(img.dtype.itemsize*8)
        img = img/255.

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        size_pixel = 2**(mask.dtype.itemsize*8)
        mask = mask/255.
        mask = mask > 0.5

        return np.transpose(img, (2, 0, 1)).astype(np.float32), np.expand_dims(mask, axis=0).astype(np.float32)

    def __len__(self):
        return self.data_len

    def _apply_contrast(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Define CLAHE
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # Apply CLAHE on Y-channel
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)  # Convert back to RGB
        return img

class DataTestModel(Dataset):
    def __init__(self, img_size, DATASET_PATH, apply_contrast=False):
        self.DATASET_PATH = DATASET_PATH
        self.images_name = np.array(list(os.listdir(self.DATASET_PATH)))
        self.data_len = len(self.images_name)
        self.img_size = img_size
        self.apply_contrast = apply_contrast

        indices = np.arange(len(self.images_name))
        np.random.shuffle(indices)
        self.images_name = self.images_name[indices]

    def __getitem__(self, item):
        img_path = os.path.join(self.DATASET_PATH, self.images_name[item])

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.apply_contrast:
            img = self._apply_contrast(img)
        img = img / 255.
        return np.transpose(img, (2, 0, 1)).astype(np.float32)

    def __len__(self):
        return self.data_len

    def _apply_contrast(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img


if __name__ == "__main__":
    pass
    # data_test = DataTestModel(384, DATASET_PATH="../dataset/Test", apply_contrast=True)
    # # ds = DatasetSegWater('train', 0.7, 256, shuffle=True, DATASET_PATH='./dataset/Water Bodies Dataset', apply_contrast=True)
    # #
    # for i in range(len(data_test)):
    #     input_torch = data_test[i]
    #     utils.show_img(input_torch.transpose(1, 2, 0))
    #     plt.show()
