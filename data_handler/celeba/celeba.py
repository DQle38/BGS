import torch
from os.path import join
import pandas
import numpy as np
from functools import partial
from torchvision.datasets.utils import verify_str_arg
import data_handler.celeba.celeba_generic as celeba_generic
import itertools


class CelebA(celeba_generic.CelebAGeneric):
    target_attr = 'Blond_Hair'
    task_split_attrs = ['Young', 'Smiling', 'Straight_Hair']

    def __init__(self, task_id, root='./data/celeba', split='train', seed=0):

        super(CelebA, self).__init__(root, split, seed)

        task_samples = self._get_task_samples(task_id, split='train')
        self.features, self.labels, self.groups = self._extract_data_per_task(task_samples, task_id=task_id)
        self.data_count()

        if split == 'test':
            task_samples = self._get_task_samples(task_id, split='test')
            self.features, self.labels, self.groups = self._extract_data_per_task(task_samples, task_id=task_id)
            self.data_count()

        print(f'<{self.split}> Task ID : ', task_id)

    @staticmethod
    def get_task_info():
        ntask = 8
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

        # mask = slice(None) if _split is None else (splits[1] == _split)
        if split =='test':
            mask = (splits[1] != 0)
        else:
            mask = slice(None) if _split is None else (splits[1] == _split)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div((self.attr + 1), 2, rounding_mode='floor')
        self.attr_names = list(attr.columns)

        target_idx = self.attr_names.index(self.target_attr)
        group_idx = self.attr_names.index('Male')
        task_split_ids = list(itertools.product([0, 1], repeat=len(self.task_split_attrs)))

        task_samples = {}
        n_samples_per_task = []
        for t, ts_id in enumerate(task_split_ids):
            mask = torch.ones(len(self.attr)).bool()
            for i, attr_value in enumerate(ts_id):
                attr_idx = self.attr_names.index(self.task_split_attrs[i])
                mask *= self.attr[:, attr_idx] == attr_value

            targets = self.attr[mask, target_idx]
            groups = self.attr[mask, group_idx]
            filenames = self.filename[mask]
            task_samples[t] = np.transpose(np.stack((filenames, targets, groups)))
            n_samples_per_task.append(len(targets))

        self.taskcla, self.ntask, self.num_total_classes, self.num_groups = self.get_task_info()
        self.task_id = task_id
        self.num_classes = self.taskcla[task_id][1]
        self.num_total_data = sum([len(t_data) for t, t_data in task_samples.items()])

        return task_samples
