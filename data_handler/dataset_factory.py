from data_handler.dataset import make_continual_datasets
from importlib import import_module


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def _get_source_code_name(name):
        source_code_name = {
            'split_cifar_100s': 'SplitCifar100S',
            'celeba_two': 'CelebASkew',
            'celeba': 'CelebA',
            'split_imagenet_100c': 'SplitImageNet100C',
        }
        assert name in source_code_name.keys()
        return source_code_name[name]

    @staticmethod
    def get_continual_datasets(name, seed=0, shuffle_task=True, **kwargs):

        class_name = DatasetFactory._get_source_code_name(name)
        if 'cifar' in name:
            dataset = getattr(import_module(f'data_handler.cifar100.{name}'), class_name)
        elif name.startswith('celeba'):
            dataset = getattr(import_module(f'data_handler.celeba.{name}'), class_name)
        elif 'imagenet' in name:
            dataset = getattr(import_module(f'data_handler.imagenet.{name}'), class_name)
        else:
            raise NotImplementedError()

        taskcla, _, _, num_groups = dataset.get_task_info()

        print('\nTask info =', taskcla)
        print(kwargs)
        # make dataloaders for each task first
        train_datasets, skew_ratios = make_continual_datasets(taskcla, dataset, split='train', seed=seed,
                                                              shuffle_task=shuffle_task, **kwargs)
        test_datasets, _ = make_continual_datasets(taskcla, dataset, split='test', seed=seed,
                                                   shuffle_task=shuffle_task, **kwargs)

        if name == 'split_imagenet_100c':
            for t in range(len(test_datasets)):
                test_datasets[t].biased_class = train_datasets[t].biased_class
                test_datasets[t].biased_skew_ratio = train_datasets[t].biased_skew_ratio

        return taskcla, num_groups, train_datasets, test_datasets, skew_ratios