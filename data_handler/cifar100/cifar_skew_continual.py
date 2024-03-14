from torchvision import transforms
from data_handler.dataset import GenericDataset

import numpy as np


def rgb_to_grayscale(img):
    """Convert image to gray scale"""
    pil_gray_img = img.convert('L')
    np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
    np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])

    return np_gray_img


class CifarSkewContinual(GenericDataset):
    def __init__(self, task_num, root, split='train', seed=0,
                 skew_ratio=0.8):
        super(CifarSkewContinual, self).__init__(root)

        self.seed = seed
        self.pc_valid = 0.10
        self.root = root
        self.task_num = task_num
        self.taskcla, self.ntask, self.num_classes, self.num_groups = self.get_task_info()
        self.skew_ratio = skew_ratio
        self.features, self.labels, self.groups = self._extract_data_per_task(task_num, skew_ratio, split=split)

        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])

        transform_list = [transforms.ToPILImage()]
        if split == 'train':
            # data augmentation for train
            transform_list.extend([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip()])
        transform_list.extend([transforms.ToTensor(),
                              normalize])
        self.transform = transforms.Compose(transform_list)

        print('Task ID : ', task_num)
        self.data_count()

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def get_task_info():
        raise NotImplementedError()

    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]
        group = self.groups[idx]

        image_not_aug = image
        image_not_aug = transforms.ToTensor()(image_not_aug)
        if self.transform:
            image = self.transform(image)

        return image, np.int64(group), np.int64(label), image_not_aug, idx

    def _extract_data_per_task(self, task_num, skew_ratio, split='train'):
        raise NotImplementedError()
