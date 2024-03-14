from os.path import join
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from torchvision import transforms
import data_handler
import os
from utils import list_dir, list_files
from data_handler.imagenet.corruption_util import gaussian_noise, frost


def get_noise(noise_type):
    if noise_type == 'gaussian':
        return gaussian_noise
    elif noise_type == 'frost':
        return frost
    else:
        raise NotImplementedError


class ImageNet100CGeneric(data_handler.GenericDataset):
    def __init__(self, task_id, root='./data/imagenet100', split='train', seed=0, noise_type='gaussian'):

        super(ImageNet100CGeneric, self).__init__(root, split)
        self.seed = seed
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_size = 224
        self.train_transform = transforms.Compose(
            [transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        self.test_transform = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean, std=std)])

        self.transform = self.train_transform if split == 'train' else self.test_transform

        self.noise_type = noise_type
        self.taskcla, self.ntask, self.num_total_classes, self.num_groups = self.get_task_info()
        self.task_id = task_id
        self.num_classes = 10
        self.num_total_data = 130000
        self._make_data(root, noise_type)

        split_folder = f'train_{noise_type}' if split == 'train' else f'val_{noise_type}'
        self.target_folder = join(root, split_folder)
        self._classes = sorted(list_dir(self.target_folder))
        self._classes = shuffle(self._classes, random_state=0)

    @staticmethod
    def get_task_info():
        ntask = 10
        class_per_task = 10
        num_classes = class_per_task
        num_groups = 2
        taskcla = []

        for t in range(ntask):
            taskcla.append((t, class_per_task))

        return taskcla, ntask, num_classes, num_groups

    @staticmethod
    def _make_data(root, noise_type='gaussian'):
        resize_size = 256
        resizing = transforms.Compose([
            transforms.Resize((resize_size, resize_size))
        ])
        corruption = transforms.Compose([
            transforms.Resize((resize_size, resize_size)), get_noise(noise_type)
        ])

        for split in ['train', 'val']:
            path = join(root, f'{split}_{noise_type}')
            if not os.path.isdir(path):
                target_folder = join(root, split)
                _classes = sorted(list_dir(target_folder))
                _classes_images = [
                    [(image, idx) for image in list_files(join(target_folder, c), ".JPEG")]
                    for idx, c in enumerate(_classes)
                ]
                _flat_classes_images = sum(_classes_images, [])

                for image_name, image_class in _flat_classes_images:
                    image_path = join(target_folder, _classes[image_class], image_name)
                    image = Image.open(image_path, mode="r").convert("RGB")
                    resized_image = resizing(image)
                    corrupted_image = corruption(image)

                    new_class_path = join(root, f'{split}_{noise_type}', _classes[image_class])
                    new_class_original_image_path = join(new_class_path, 'original')
                    os.makedirs(new_class_original_image_path, exist_ok=True)
                    new_class_corrupted_image_path = join(new_class_path, 'corrupted')
                    os.makedirs(new_class_corrupted_image_path, exist_ok=True)

                    image_save_path = join(new_class_original_image_path, image_name)
                    resized_image.save(image_save_path)
                    cor_image_save_path = join(new_class_corrupted_image_path, image_name)
                    corrupted_image.save(cor_image_save_path)

    def __getitem__(self, index):

        image_path = self.features[index]
        group = self.groups[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        image_not_aug = image_path

        if self.transform:
            image = self.transform(image).float()

        return image, np.int64(group), np.int64(label), image_not_aug, index