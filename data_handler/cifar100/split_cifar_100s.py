import numpy as np
from data_handler.cifar100.cifar100 import CIFAR100
from data_handler.cifar100.cifar_skew_continual import CifarSkewContinual, rgb_to_grayscale
from sklearn.utils import shuffle


class SplitCifar100S(CifarSkewContinual):
    def __init__(self, task_id, root='./data/cifar100s_task/',
                 split='train', seed=0, skew_ratio=0.5):
        super(SplitCifar100S, self).__init__(task_id, root, split, seed, skew_ratio)

    @staticmethod
    def get_task_info():
        ntask = 10
        num_total_classes = 100
        num_groups = 2
        taskcla = []
        class_per_task = num_total_classes // ntask # Task Incremental Setting
        num_classes = class_per_task
        for t in range(ntask):
            taskcla.append((t, class_per_task))

        return taskcla, ntask, num_classes, num_groups

    def _extract_data_per_task(self, task_num, skew_ratio, split='train'):

        train = True if split == 'train' else False
        cifar100_data = CIFAR100('./data/', train=train, download=True)

        unique_labels = np.unique(cifar100_data.fine_targets)
        class_ids = list(shuffle(unique_labels, random_state=0))
        cur_task_label = class_ids[(task_num) * self.num_classes: (task_num + 1) * self.num_classes]
        print('Task classes : ', cur_task_label)

        self.num_total_data = len(cifar100_data)
        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)  # ngroups * nclasses
        num_data_per_class = int(len(cifar100_data) / cifar100_data.num_classes)
        num_skewed_data_per_class = int(num_data_per_class * skew_ratio)  # 500 * skew_ratio per class per task

        task_features, task_labels, task_groups = [], [], []
        for i, (img, fine_label, coarse_label, _, _) in enumerate(cifar100_data):
            if int(fine_label) in cur_task_label:
                label = cur_task_label.index(int(fine_label))
                if not train:
                    gray_image = rgb_to_grayscale(img)
                    color_image = np.array(img)
                    task_features.extend([gray_image, color_image])
                    task_labels.extend([label, label])
                    task_groups.extend([0, 1])
                else:
                    num_data_for_gray = num_skewed_data_per_class if label < 5 \
                        else num_data_per_class - num_skewed_data_per_class

                    group_of_image = 0 if data_count[0, label] < num_data_for_gray else 1

                    if group_of_image == 0:
                        skewed_image = rgb_to_grayscale(img)
                    else:
                        skewed_image = np.array(img)

                    data_count[group_of_image, label] += 1
                    task_features.append(skewed_image)
                    task_labels.append(label)
                    task_groups.append(group_of_image)

        return np.stack(task_features), np.array(task_labels, dtype=int), np.array(task_groups, dtype=int)

