import os
import PIL
import numpy as np
import zipfile
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from torchvision import transforms
import data_handler


class CelebAGeneric(data_handler.GenericDataset):
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    mean = [0.5063, 0.4258, 0.3832]
    std = [0.3093, 0.2890, 0.2884]

    train_transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)]
    )
    name = 'celeba'


    def __init__(self, root='./data/celeba', split='train', seed=0):

        super(CelebAGeneric, self).__init__(root, split)
        self.seed = seed
        self.transform = self.train_transform if split == 'train' else self.test_transform

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.sensitive_attr = 'Male'


    @staticmethod
    def get_task_info():
        pass

    def _extract_data_per_task(self, feature_mat, task_id):
        features, labels, groups = [], [], []

        for rows in feature_mat[task_id]:
            filename, t, g = rows
            features.append(filename)
            labels.append(t)
            groups.append(g)
        return features, np.array(labels, dtype=int), np.array(groups, dtype=int)

    def __getitem__(self, index):
        img_name = self.features[index]
        group = self.groups[index]
        label = self.labels[index]
        image_path = os.path.join(self.root, "img_align_celeba", img_name)
        image = PIL.Image.open(image_path)
        image_not_aug = image_path

        if self.transform:
            image = self.transform(image)

        return image, np.int64(group), np.int64(label), image_not_aug, index

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, "img_align_celeba"))

    def _download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, self.root, filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, "img_align_celeba.zip"), "r") as f:
            f.extractall(self.root)

    def _shuffle_data(self, features, labels, groups):
        num_data = len(labels)
        np.random.seed(self.seed)
        shuffled_idxs = np.arange(num_data, dtype=np.int32)
        np.random.shuffle(shuffled_idxs)
        # print(shuffled_idxs)
        features = np.array(features)[shuffled_idxs]
        labels = np.array(labels)[shuffled_idxs]
        groups = np.array(groups)[shuffled_idxs]
        return features, labels, groups