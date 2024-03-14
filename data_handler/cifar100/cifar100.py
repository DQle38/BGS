from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR100(VisionDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This a CIFAR100 with superclasses version.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': ['fine_label_names', 'coarse_label_names'],
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            seed: int = 0,
            shuffle = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,

    ) -> None:

        super(CIFAR100, self).__init__(root, transform=transform,
                                       target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.fine_targets = []
        self.coarse_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.fine_targets.extend(entry['fine_labels'])
                self.coarse_targets.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if shuffle:
            np.random.seed(seed)
            idx = np.arange(len(self.data), dtype=np.int64)
            np.random.shuffle(idx)
            self.data = self.data[idx]
            #             print(idx.shape)
            #             print(self.targets.shape)
            self.fine_targets = np.array(self.fine_targets)[idx]
            self.coarse_targets = np.array(self.coarse_targets)[idx]

        self.num_groups = 1
        self.num_classes = 100
        self.labelwise=False

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        for i, key in enumerate(self.meta['key']):
            with open(path, 'rb') as infile:
                data = pickle.load(infile, encoding='latin1')
                if i == 0:
                    self.fine_classes = data[key]
                else:
                    self.coarse_classes = data[key]

        self.fine_class_to_idx = {_class: i for i, _class in enumerate(self.fine_classes)}
        self.coarse_class_to_idx = {_class: i for i, _class in enumerate(self.coarse_classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, fine_target, coarse_target) where target is index of the target class.
        """
        img, fine_target, coarse_target = self.data[index], self.fine_targets[index], self.coarse_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            fine_target = self.target_transform(fine_target)
            coarse_target = self.target_transform(coarse_target)

        return img, fine_target, coarse_target, 0, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


