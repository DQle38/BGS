import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from data_handler.imagenet.corruption_util import gaussian_noise, frost


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer(data.Dataset):
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, data, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'groups']
        self.data = data
        self.transform = get_transform(data)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor,
                     task_labels: torch.Tensor, groups: torch.Tensor, continual='task', num_total_classes=100) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param groups: tensor containing the group labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                if attr_str == 'examples' and self.data in ['split_cifar_100s']:
                    typ = examples.dtype
                elif attr_str.endswith('bels'):
                    typ = torch.int64
                else:
                    typ = torch.float32

                if attr_str == 'examples' and self.data not in ['split_cifar_100s']:
                    setattr(self, attr_str, np.empty((self.buffer_size,), dtype=object))
                elif attr_str == 'logits':
                    if continual == 'class':
                        setattr(self, attr_str, torch.zeros((self.buffer_size, num_total_classes), dtype=typ))
                    else:
                        setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ))
                else:
                    # setattr(self, attr_str, -1 * torch.ones((self.buffer_size, *attr.shape[1:]), dtype=typ))
                    setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, groups=None,
                 continual='task', num_total_classes=100):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param groups: tensor containing the group labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, groups, continual, num_total_classes)

        for i in range(len(examples)):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i]
                if labels is not None:
                    # self.labels[index] = labels[i].to(self.device)
                    self.labels[index] = labels[i]
                if logits is not None:
                    self.logits[index] = logits[i]
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i]
                if groups is not None:
                    self.groups[index] = groups[i]

    def __len__(self):
        return min(self.num_seen_examples, len(self.examples))

    def __getitem__(self, idx):
        if self.data not in ['split_cifar_100s']:
            img = Image.open(self.examples[idx]).convert('RGB')
        else:
            img = self.examples[idx]

        ret_tuple = (img if self.transform is None else self.transform(img),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[idx],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


def get_noise(noise_type):
    if noise_type == 'gaussian':
        return gaussian_noise
    elif noise_type == 'frost':
        return frost
    else:
        raise NotImplementedError


def get_transform(dataset):
    if dataset == 'split_cifar_100s':
        mean=[0.5071, 0.4865, 0.4409]
        std=[0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])

    elif dataset == 'split_imagenet_100c':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_size = 224
        transform = transforms.Compose(
            [transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    elif dataset.startswith('celeba'):
        mean = [0.5063, 0.4258, 0.3832]
        std = [0.3093, 0.2890, 0.2884]
        transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)]
        )

    else:
        raise NotImplementedError

    return transform


class BufferGDumb(Buffer):
    def __init__(self, buffer_size, device, data, n_tasks=None):
        self.count_dict = np.zeros(100)
        self.idx_per_class = {}
        self.num_classes = 0
        self.cur_num_classes = 0
        super().__init__(buffer_size, device, data, n_tasks)
        print('buffer : ', self.buffer_size)

    def gdumb(self, label: int) -> int:
         # if the size of the given label is already full
        if self.count_dict[label] >= self.num_data_per_class:
            return -1
        
        # if the size of the buffer is not full yet,
        if self.count_dict.sum() < self.buffer_size:
            idx = int(self.count_dict.sum())
            return idx
        
        else:
            max_labels = np.where(self.count_dict==self.count_dict.max())[0]
            pop_label = np.random.choice(max_labels)
            idx = np.random.randint(self.count_dict[pop_label])

            # pop the existing data
            index = self.idx_per_class[pop_label].pop(idx)

            self.count_dict[pop_label] -= 1
            return index

    def add_new_classes(self, num_classes):
        for i in range(self.num_classes, self.num_classes+num_classes):
            self.idx_per_class[i] = []
        self.cur_num_classes = num_classes
        self.num_classes += num_classes
        self.num_data_per_class = self.buffer_size // self.num_classes 
    
    def add_data(self, examples, labels=None, logits=None, task_labels=None, groups=None,
                 continual='task', num_total_classes=100):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, groups, continual, num_total_classes)
        
        cumul_labels = labels+self.num_classes-self.cur_num_classes
        for i in range(len(examples)):
            index = self.gdumb(cumul_labels[i].item())

            if index >= 0:
                self.num_seen_examples += 1
                self.examples[index] = examples[i]
                if labels is not None:
                    # self.labels[index] = labels[i].to(self.device)
                    self.labels[index] = labels[i]
                if logits is not None:
                    self.logits[index] = logits[i]
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i]
                if groups is not None:
                    self.groups[index] = groups[i]

                self.count_dict[cumul_labels[i].item()] += 1
                self.idx_per_class[cumul_labels[i].item()].append(index)

            num_new_data = self.count_dict[self.num_classes-self.cur_num_classes:self.num_classes].sum()
            if num_new_data == self.num_data_per_class * self.cur_num_classes:
                return True
            
        return False


class BufferGDumbBalanced(Buffer):
    def __init__(self, buffer_size, device, data, n_tasks=None):
        self.num_classes = 0
        self.cur_num_classes = 0
        self.num_groups = 2

        self.idx_per_group_class = {}
        self.count_dict = np.zeros((self.num_groups, 100))
        for g in range(self.num_groups):
            self.idx_per_group_class[g] = {}
        
        super().__init__(buffer_size, device, data, n_tasks)

    def gdumb_balanced(self, label: int, group: int) -> int:
         # if the size of the given label is already full
        if self.count_dict[group][label] >= self.num_data_per_group_class:
            return -1
        
        # if the size of the buffer is not full yet,

        if self.count_dict.sum() < self.buffer_size:
            idx = int(self.count_dict.sum())
            return idx
        
        else:
            max_groups, max_labels = np.where(self.count_dict == self.count_dict.max())
            pop_idx = np.random.randint(max_labels.shape[0])
            pop_group, pop_label = max_groups[pop_idx], max_labels[pop_idx]
            idx = np.random.randint(self.count_dict[pop_group][pop_label])
            
            # pop the existing data
            index = self.idx_per_group_class[pop_group][pop_label].pop(idx)
            self.count_dict[pop_group][pop_label] -= 1
            return index

    def add_new_classes(self, num_classes):
        for g in range(self.num_groups):
            for i in range(self.num_classes, self.num_classes+num_classes):
                self.idx_per_group_class[g][i] = []
        self.cur_num_classes = num_classes
        self.num_classes += num_classes
        self.num_data_per_group_class = self.buffer_size // (self.num_classes * self.num_groups)
    
    def add_data(self, examples, labels=None, logits=None, task_labels=None, groups=None,
                 continual='task', num_total_classes=100):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, groups, continual, num_total_classes)
        
        cumul_labels = labels+self.num_classes-self.cur_num_classes
        for i in range(len(examples)):
            index = self.gdumb_balanced(cumul_labels[i].item(), groups[i].item())
            if index >= 0:
                self.num_seen_examples += 1
                self.examples[index] = examples[i]
                if labels is not None:
                    self.labels[index] = labels[i]
                if logits is not None:
                    self.logits[index] = logits[i]
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i]
                if groups is not None:
                    self.groups[index] = groups[i]

                self.count_dict[groups[i].item()][cumul_labels[i].item()] += 1
                self.idx_per_group_class[groups[i].item()][cumul_labels[i].item()].append(index)
            
            num_new_data = 0
            for _g in range(self.num_groups):
                num_new_data += self.count_dict[_g][self.num_classes-self.cur_num_classes:self.num_classes].sum()
            if num_new_data == self.num_data_per_group_class * (self.cur_num_classes*self.num_groups):
                return True
            
        return False
    