from abc import ABC

import torch.utils.data as data
import numpy as np
from sklearn.utils import shuffle

bias_levels = [0.5, 0.6826, 0.8223, 0.9087, 0.9554, 0.9787, 0.99]


def make_continual_datasets(taskcla, dataset, seed=0, split='train', shuffle_task=True, task_ids=None, **kwargs):

    task_datasets = []
    n_tasks = len(taskcla) if kwargs['n_tasks'] is None else kwargs.pop('n_tasks')
    start_task = kwargs.pop('start_task')
    num_task_needed = min(start_task + n_tasks, len(taskcla))

    if task_ids is None:
        if shuffle_task:
            task_ids = list(shuffle(np.arange(len(taskcla)), random_state=seed))
        else:
            task_ids = list(np.arange(len(taskcla)))

    print('Task ID: ', task_ids)

    skew_ratios = kwargs.pop('skew_ratio') if 'skew_ratio' in kwargs.keys() else None
    for_backward = kwargs.pop('for_backward') if 'for_backward' in kwargs.keys() else None
    _biased = kwargs.pop('biased') if 'biased' in kwargs.keys() else None
    noise_type = kwargs.pop('noise_type') if 'noise_type' in kwargs.keys() else None
    accumulation = kwargs.pop('accumulation') if 'accumulation' in kwargs.keys() else None
    n_biased = kwargs.pop('n_biased') if 'n_biased' in kwargs.keys() else None
    for_two = True if num_task_needed == 2 else False

    if dataset.__name__ in ['SplitCifar100S', 'CelebASkew', 'SplitImageNet100C']:
        skew_ratios = make_skew_ratio_list(skew_ratios, task_ids, n_tasks if accumulation else len(taskcla),
                                           dataset.__name__, seed=0, accumulation=accumulation,
                                           n_biased=n_biased, for_backward=for_backward)

        skew_ratios = np.array(skew_ratios) if accumulation else np.array(skew_ratios)[np.array(task_ids)[:num_task_needed]]

    for t in range(num_task_needed):
        if dataset.__name__ in ['SplitCifar100S', 'CelebASkew', 'SplitImageNet100C']:
            skew_ratio = skew_ratios[t]
            if dataset.__name__ == 'SplitImageNet100C':
                task_datasets.append(dataset(task_id=task_ids[t], split=split, seed=seed,
                                             skew_ratio=skew_ratio, noise_type=noise_type, for_two=for_two, **kwargs))
            else:
                task_datasets.append(dataset(task_id=task_ids[t], split=split, seed=seed, skew_ratio=skew_ratio, **kwargs))
        else:
            task_datasets.append(dataset(task_id=task_ids[t], split=split, seed=seed, **kwargs))

    return task_datasets, skew_ratios


def make_skew_ratio_list(skew_ratios, task_ids, n_tasks, dataset_name, seed=0,
                         accumulation=False, n_biased=0, for_backward=False):
    np.random.seed(seed)
    skew_ratio_list = []
    if accumulation:
        assert n_biased < n_tasks
        random_idx = np.random.permutation(n_tasks-1)
        if for_backward:
            random_idx = random_idx + 1
        for i in range(n_tasks):
            skew_ratio_list.append(0.5)
        for i in range(n_biased):
            skew_ratio_list[random_idx[i]] = 0.9554
    else:

        for i in range(n_tasks):
            if dataset_name == 'SplitImageNet100C':
                ratio = np.random.uniform(low=0.5, high=0.95)
            else:
                ratio = np.random.uniform(low=0.5, high=0.99)
            skew_ratio_list.append(ratio)

        if skew_ratios is not None:
            assert len(skew_ratios) <= n_tasks
            for t, skew in enumerate(skew_ratios):
                skew_ratio_list[task_ids[t]] = skew # append pre-defined skew ratios to the sequence

    return skew_ratio_list


class GenericDataset(data.Dataset, ABC):
    def __init__(self, root, split='train'):
        # TODO: continual setting
        self.root = root
        self.split = split
        self.num_data = None
        self.task_num = None
        self.labels = None
        self.groups = None
        self.features = None
        self.num_groups = None
        self.num_classes = None
        self.class_per_task = None

    def __len__(self):
        return int(np.sum(self.num_data))

    def make_weights(self):
        group_weights = len(self) / self.num_data
        weights = [group_weights[int(self.groups[i]), int(y)] for i, y in enumerate(self.labels)]
        return weights

    def data_count(self, cur_task_num_classes=None):
        cur_task_num_classes = self.num_classes if cur_task_num_classes is None else cur_task_num_classes

        num_data = np.zeros((self.num_groups, cur_task_num_classes))
        for i in range(len(self.labels)):
            
            group = self.groups[i]
            label = self.labels[i]
            num_data[group, label] += 1
            
        print('Data Count')
        print(num_data)
        self.num_data = num_data
