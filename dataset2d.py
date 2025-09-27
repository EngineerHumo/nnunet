import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


ia.seed(1)
seq = iaa.Sequential([
    iaa.Sharpen((0.0, 1.0)),
    iaa.Affine(scale=(1, 2)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.1))
], random_order=True)


def to_one_hot(label, num_classes):
    """Convert a 2D label map to a (C, H, W) one-hot representation."""
    h, w = label.shape
    one_hot = np.zeros((num_classes, h, w), dtype=np.uint8)
    for class_idx in range(num_classes):
        one_hot[class_idx] = (label == class_idx).astype(np.uint8)
    return one_hot


class Data(data.Dataset):
    def __init__(self, base_dir='./data/', train=True, dataset='ISIC16', crop_szie=None):
        super(Data, self).__init__()
        self.dataset_dir = base_dir
        self.train = train
        self.dataset = dataset
        self.images = []
        self.labels = []
        self.names = []

        if crop_szie is None:
            if self.dataset == 'prp':
                crop_szie = [1240, 1240]
            else:
                crop_szie = [512, 512]
        self.crop_size = crop_szie

        if self.dataset in ['acdc', 'synapse']:
            if train:
                data_dir = os.path.join(self.dataset_dir, self.dataset, 'data_npz')
                txt = os.path.join(self.dataset_dir, self.dataset, 'annotations', 'train.txt')
            else:
                data_dir = os.path.join(self.dataset_dir, self.dataset, 'data_npz')
                txt = os.path.join(self.dataset_dir, self.dataset, 'annotations', 'test.txt')

            with open(txt, 'r') as f:
                filename_list = [line.strip() for line in f.readlines()]

            for filename in filename_list:
                npz_path = os.path.join(data_dir, filename + '.npz')
                data_npz = np.load(npz_path)
                image, label = data_npz['image'], data_npz['label']

                image = np.array(image)
                label = np.array(label)

                if not self.train:
                    image = cv2.resize(image, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
                    image = np.expand_dims(image, axis=2)
                    label = cv2.resize(label, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)

                self.images.append(image)
                self.labels.append(label)
                self.names.append(filename)

        elif self.dataset in ['ISIC16', 'ISIC18', 'prp']:
            if train:
                image_dir = os.path.join(self.dataset_dir, self.dataset, 'images')
                label_dir = os.path.join(self.dataset_dir, self.dataset, 'labels')
                txt = os.path.join(self.dataset_dir, self.dataset, 'annotations', 'train.txt')
            else:
                image_dir = os.path.join(self.dataset_dir, self.dataset, 'images')
                label_dir = os.path.join(self.dataset_dir, self.dataset, 'labels')
                txt = os.path.join(self.dataset_dir, self.dataset, 'annotations', 'test.txt')

            with open(txt, 'r') as f:
                filename_list = [line.strip() for line in f.readlines()]

            for filename in filename_list:
                if self.dataset == 'prp':
                    image_path = os.path.join(image_dir, filename + '.png')
                    label_path = os.path.join(label_dir, filename + '.png')
                else:
                    image_path = os.path.join(image_dir, filename + '.jpg')
                    if self.dataset == 'ISIC16':
                        label_path = os.path.join(label_dir, filename + '.png')
                    else:
                        label_path = os.path.join(label_dir, filename + '_segmentation.png')

                image = Image.open(image_path).convert('RGB')
                label = Image.open(label_path)

                image = np.array(image)
                label = np.array(label)

                if not self.train:
                    image = cv2.resize(image, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
                    label = cv2.resize(label, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
                    if self.dataset in ['ISIC16', 'ISIC18']:
                        label = (label / 255).astype(np.uint8)

                self.images.append(image)
                self.labels.append(label)
                self.names.append(filename)
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset}')

        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(self.images[index], copy=True)
        label = np.array(self.labels[index], copy=True)


        sample = {'image': image, 'label': label, 'name': self.names[index]}



        if self.dataset in ['acdc', 'synapse']:
            sample['label'] = sample['label'].astype(np.int16)

        prob = random.random()
        if self.train and prob > 0.5:
            segmap = SegmentationMapsOnImage(sample['label'], shape=sample['image'].shape)
            aug_image, aug_label = seq(image=sample['image'], segmentation_maps=segmap)
            sample['image'] = aug_image
            sample['label'] = aug_label.get_arr()

        if self.train:
            sample['image'] = cv2.resize(sample['image'], (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
            if self.dataset in ['acdc', 'synapse']:
                sample['image'] = np.expand_dims(sample['image'], axis=2)
            sample['label'] = cv2.resize(sample['label'], (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)

        if self.dataset in ['ISIC16', 'ISIC18']:
            sample['label'] = (sample['label'] / 255).astype(np.uint8)

        label_indices = sample['label'].astype(np.int64)
        if label_indices.ndim == 3 and label_indices.shape[2] == 1:
            label_indices = np.squeeze(label_indices, axis=2)

        if self.dataset == 'prp':
            # Convert the label map (H, W) into four stacked binary maps (4, H, W).
            one_hot_label = to_one_hot(label_indices, 4).astype(np.float32)
            sample['label_indices'] = label_indices.astype(np.int64)
            sample['label'] = one_hot_label
        else:
            if label_indices.ndim == 2:
                expanded = np.expand_dims(label_indices, axis=0)
            else:
                expanded = label_indices
            sample['label_indices'] = label_indices.astype(np.int64)
            sample['label'] = expanded.astype(np.float32)

        image_array = sample['image'].astype(np.float32)
        if image_array.ndim == 3:
            image_array = np.transpose(image_array, (2, 0, 1))
        else:
            image_array = np.expand_dims(image_array, axis=0)
        sample['image'] = image_array

        return sample

    def __str__(self):
        return 'dataset:{} train:{}'.format(self.dataset, self.train)
