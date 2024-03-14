import torch
from os.path import join
import pandas
import numpy as np
from functools import partial
from torchvision.datasets.utils import verify_str_arg
import data_handler.celeba.celeba_generic as celeba_generic


class CelebASkew(celeba_generic.CelebAGeneric):
    def __init__(self, task_id, root='./data/celeba', split='train', seed=0, skew_ratio=0.5):

        super(CelebASkew, self).__init__(root, split, seed)
        # SELECT the features
        task_samples = self._get_task_samples(task_id, split=split)
        self.features, self.labels, self.groups = self._extract_data_per_task(task_samples, task_id=task_id)
        self.skew_ratio = skew_ratio
        print(f'<{self.split}> Task ID : ', task_id)
        print('Before skew =====')
        self.data_count()
        if self.split == 'train':
            self.features, self.labels, self.groups = self.skew_data(skew_ratio)
        else:
            self.features, self.labels, self.groups = self.skew_data(skew_ratio=0.5)

        print('After skew =====')
        self.data_count()

    @staticmethod
    def get_task_info():
        ntask = 2
        class_per_task = 2
        num_classes = class_per_task
        num_groups = 2
        taskcla = []

        for t in range(ntask):
            taskcla.append((t, class_per_task))

        return taskcla, ntask, num_classes, num_groups

    def _get_task_samples(self, task_id, split):
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        _split = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "valid", "test", "all"))]
        fn = partial(join, self.root)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if _split is None else (splits[1] == _split)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div((self.attr + 1), 2, rounding_mode='floor')
        self.attr_names = list(attr.columns)
        task_split_attr = 'Mouth_Slightly_Open'
        target_attr = 'Young'
        self.target_attr = target_attr

        target_idx = self.attr_names.index(self.target_attr)
        group_idx = self.attr_names.index('Male')
        task_split_idx = self.attr_names.index(task_split_attr)

        task_samples = {}
        n_samples_per_task = []
        for t in range(2):
            mask = torch.ones(len(self.attr)).bool()
            mask *= self.attr[:, task_split_idx] == t

            targets = self.attr[mask, target_idx]
            groups = self.attr[mask, group_idx]
            filenames = self.filename[mask]
            task_samples[t] = np.transpose(np.stack((filenames, targets, groups)))
            n_samples_per_task.append(len(targets))

        self.taskcla, self.ntask, self.num_total_classes, self.num_groups = self.get_task_info()
        self.task_id = task_id
        self.num_classes = self.taskcla[task_id][1]
        self.num_total_data = 20000

        return task_samples

    def skew_data(self, skew_ratio):
        features, labels, groups = [], [], []
        n_data_per_class = 5000 if self.split == 'train' else 500
        n_skewed_for_major = int(n_data_per_class * skew_ratio)
        n_skewed_for_minor = n_data_per_class - n_skewed_for_major
        num_data = np.array([[n_skewed_for_minor, n_skewed_for_major],
                             [n_skewed_for_major, n_skewed_for_minor]], dtype=int)

        self.features, self.labels, self.groups = self._shuffle_data(self.features, self.labels, self.groups)
        data_count = np.zeros((self.num_groups, self.num_classes))
        for i in range(len(self.labels)):
            if data_count[self.groups[i], self.labels[i]] < num_data[self.groups[i], self.labels[i]]:
                features.append(self.features[i])
                labels.append(self.labels[i])
                groups.append(self.groups[i])
            data_count[self.groups[i], self.labels[i]] += 1

        return features, np.array(labels, dtype=int), np.array(groups, dtype=int)
