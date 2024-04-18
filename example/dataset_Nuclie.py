import os
import cv2
import numpy as np
import surkit.backend as bkd

if os.environ['SRK_BACKEND'] == 'jax':
    from jax_dataloader import Dataset
elif os.environ['SRK_BACKEND'] == 'pytorch':
    from torch.utils.data import Dataset
elif os.environ['SRK_BACKEND'] == 'oneflow':
    from oneflow.utils.data import Dataset

mean = 0.5
std = 0.5

def normalize(image, mean, std):
    return (image - mean * 255) / (std * 255)

def transform(image, target):
    if np.random.randn() > 0.5:
        image, target = np.fliplr(image), np.fliplr(target)
    return normalize(image, mean, std), target

def denormalize(image):
    return image * std * 255 + mean * 255

class Nuclie_data(Dataset):
    def __init__(self, path, train=True):
        self.path = path
        if train:
            self.folders = os.listdir(path)[:500]
        else:
            self.folders = os.listdir(path)[500:]
        self.size = 96, 96
            
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        if bkd.backend_name == 'jax' and type(idx) == list:
            imgs = np.zeros((len(idx), 3, *self.size))
            masks = np.zeros((len(idx), 1, *self.size))
            for i, v in enumerate(idx):
                image_folder = os.path.join(self.path, self.folders[v], 'images/')
                mask_folder = os.path.join(self.path, self.folders[v], 'masks/')
                image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
                img = cv2.imread(image_path)[:, :, :3].astype('float32')
                img = cv2.resize(img, self.size)
                mask = self.get_mask(mask_folder)

                img, mask = transform(img, mask)
                img, mask = img.transpose(2, 0, 1), mask.transpose(2, 0, 1)
                imgs[i] = img
                masks[i] = mask
            return bkd.np_to_tensor(imgs.copy()), bkd.np_to_tensor(masks.copy())
        
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
        img = cv2.imread(image_path)[:, :, :3].astype('float32')
        img = cv2.resize(img, self.size)
        mask = self.get_mask(mask_folder)
        
        img, mask = transform(img, mask)
        img, mask = img.transpose(2, 0, 1), mask.transpose(2, 0, 1)
        return bkd.np_to_tensor(img.copy()), bkd.np_to_tensor(mask.copy())

    def get_mask(self, mask_folder):
        mask = np.zeros((*self.size, 1))
        for mask_ in os.listdir(mask_folder):
            mask_ = cv2.imread(os.path.join(mask_folder,mask_))[..., 0].astype('float32')
            mask_ = cv2.resize(mask_, self.size)
            mask_ = np.expand_dims(mask_,axis=-1) != 0
            mask = np.maximum(mask, mask_)
        return mask