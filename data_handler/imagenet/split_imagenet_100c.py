from os.path import join
from utils import list_dir

import numpy as np
from utils import list_files
from data_handler.imagenet.imagenet100c_generic import ImageNet100CGeneric


class SplitImageNet100C(ImageNet100CGeneric):
    def __init__(self, task_id, root='./data/imagenet100', split='train', seed=0,
                 noise_type='gaussian', skew_ratio=0.5, for_two=False):

        super(SplitImageNet100C, self).__init__(task_id, root, split, seed, noise_type)

        self.noise_ratio = 0.01 if for_two else 0.05
        self.features, self.labels, self.groups = self._extract_data_per_task(self._classes, task_id=task_id,
                                                                              split=split, skew_ratio=skew_ratio)
        print(f'<{self.split}> Task ID : ', task_id)
        self.data_count()

    def _extract_data_per_task(self, classes, task_id, split='train', skew_ratio=0.5):
        features, labels, groups = [], [], []

        self.task_classes = classes[self.num_classes * task_id: self.num_classes * (task_id + 1)]

        _split = 'val' if split == 'test' else split
        target_folder = join(self.root, _split)
        _classes = sorted(list_dir(target_folder))
        _classes_images = [
            [(image, idx) for image in list_files(join(target_folder, c), ".JPEG")]
            for idx, c in enumerate(self.task_classes)
        ]
        _flat_classes_images = sum(_classes_images, [])

        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)  # ngroups * nclasses
        num_data_per_class = int(len(_flat_classes_images) / len(self.task_classes))
        self.biased_skew_ratio = skew_ratio
        # biased_skew_ratio = 0.99 if biased else 0.5
        num_skewed_data_per_class = int(num_data_per_class * self.biased_skew_ratio)  # 1300 * skew_ratio per class per task
        self.biased_class = np.random.randint(0, 10)

        for image_name, image_class in _flat_classes_images:
            if split == 'test':
                orig_image_path = join(self.target_folder, self.task_classes[image_class], 'original', image_name)
                cor_image_path = join(self.target_folder, self.task_classes[image_class], 'corrupted', image_name)
                features.extend([orig_image_path, cor_image_path])
                labels.extend([image_class, image_class])
                groups.extend([0, 1])
            else:
                num_data_for_corrupted = num_skewed_data_per_class if image_class == self.biased_class \
                    else int(num_data_per_class * self.noise_ratio)

                group_of_image = 1 if data_count[1, image_class] < num_data_for_corrupted else 0
                data_count[group_of_image, image_class] += 1

                target_group = 'original' if group_of_image == 0 else 'corrupted'
                image_path = join(self.target_folder, self.task_classes[image_class], target_group, image_name)
                features.append(image_path)
                labels.append(image_class)
                groups.append(group_of_image)

        return features, np.array(labels, dtype=int), np.array(groups, dtype=int)
