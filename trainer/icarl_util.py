import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from data_handler.imagenet.corruption_util import gaussian_noise, frost


class ExemplarSet(data.Dataset):
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, data, num_total_classes=100):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.num_seen_classes = 0

        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'groups']
        self.data = data
        self.transform = get_transform(data)
        self.fea_emb_transform = get_transform(data, for_embedding=True)
        self.per_class_exemplar = {c: [] for c in range(num_total_classes)}
        self.per_class_exemplar_groups = {c: [] for c in range(num_total_classes)}
        self.per_class_exemplar_features = {c: [] for c in range(num_total_classes)}

    def reduce_exemplar_set(self, num_new_classes):
        num_seen_classes = self.num_seen_classes + num_new_classes
        samples_per_class = round(self.buffer_size / num_seen_classes)
        for c in range(self.num_seen_classes):
            self.per_class_exemplar[c] = self.per_class_exemplar[c][:samples_per_class]
            self.per_class_exemplar_features[c] = self.per_class_exemplar_features[c][:samples_per_class]
            self.per_class_exemplar_groups[c] = self.per_class_exemplar_groups[c][:samples_per_class]

        self.num_seen_classes = num_seen_classes

    def construct_exemplar_set(self, model, train_dataset):
        buffer_per_class = round(self.buffer_size / self.num_seen_classes)
        train_dataset = deepcopy(train_dataset)
        train_dataset.transform = self.fea_emb_transform
        num_classes = train_dataset.num_classes
        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, drop_last=False,
                                                   shuffle=False, sampler=None, **kwargs)
        # First, get class mean embedding
        model.eval()
        with torch.no_grad():
            if self.data == 'split_cifar_100s':
                class_means = torch.zeros(num_classes, 64).cuda()
            else:
                class_means = torch.zeros(num_classes, 512).cuda()

            per_class_n_data = torch.zeros(num_classes).cuda()
            per_class_images = {c: [] for c in range(num_classes)}
            per_class_groups = {c: [] for c in range(num_classes)}
            per_class_features = {c: [] for c in range(num_classes)}
            for samples in tqdm(train_loader):
                data, group, target, data_not_aug, _ = samples
                data, group, target = data.cuda(), group.cuda(), target.cuda()

                _, features, _ = model(data, get_inter=True)

                for c in range(num_classes):
                    if (target==c).sum() == 0:
                        continue
                    class_feature = features[target == c]
                    class_means[c] += class_feature.sum(0)
                    per_class_n_data[c] += (target == c).sum()
                    per_class_features[c].append(class_feature)

                    if self.data == 'split_cifar_100s':
                        per_class_images[c].append(data_not_aug[target == c])
                    else:
                        per_class_images[c].append(np.array(data_not_aug, dtype=object)[(target == c).cpu().numpy()])
                    per_class_groups[c].append(group[target == c].cpu())

            for c in range(num_classes):
                per_class_features[c] = torch.cat(per_class_features[c])
                if self.data == 'split_cifar_100s':
                    per_class_images[c] = torch.cat(per_class_images[c])
                else:
                    per_class_images[c] = np.concatenate(per_class_images[c])
                per_class_groups[c] = torch.cat(per_class_groups[c])

            class_means /= per_class_n_data.unsqueeze(1)
            class_means = class_means / torch.norm(class_means, 2, 1).unsqueeze(1)
            class_offset = self.num_seen_classes - num_classes
            per_class_index_set = {c: [] for c in range(num_classes)}
            for c in range(num_classes):
                for k in range(buffer_per_class):
                    if k == 0:
                        distance = torch.sqrt(torch.sum((class_means[c].unsqueeze(0) - per_class_features[c])**2, dim=1))
                    else:
                        feature_sum = per_class_features[c] + \
                                      torch.cat(self.per_class_exemplar_features[class_offset+c]).sum(0).unsqueeze(0)

                        feature_sum = feature_sum / (k+1)
                        feature_sum = feature_sum / torch.norm(feature_sum, 2, 1).unsqueeze(1)
                        distance = torch.sqrt(torch.sum((class_means[c].unsqueeze(0) - feature_sum)**2, dim=1))
                    indicies = torch.argsort(distance)
                    for i in indicies:
                        if i.item() in per_class_index_set[c]:
                            continue
                        else:
                            per_class_index_set[c].append(i.item())
                            self.per_class_exemplar[class_offset+c].append(per_class_images[c][i])
                            self.per_class_exemplar_groups[class_offset+c].append(per_class_groups[c][i])
                            self.per_class_exemplar_features[class_offset+c].append(per_class_features[c][i])
                            break

        self.examples = []
        self.labels = []
        self.groups = []
        for c in range(self.num_seen_classes):
            class_images = self.per_class_exemplar[c]
            self.examples.extend(class_images)
            groups = self.per_class_exemplar_groups[c]
            self.groups.extend(groups)
            self.labels.extend([c] * len(class_images))

        print('Buffer updated.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.data == 'split_cifar_100s':
            img = self.examples[idx]
        else:
            img = Image.open(self.examples[idx]).convert('RGB')

        # img = self.examples[idx]
        group = self.groups[idx]

        img = self.transform(img)
        label = self.labels[idx]
        if self.data == 'split_cifar_100s':
            return img, np.int64(group), label, torch.zeros_like(img), idx
        else:
            return img, np.int64(group), label, '0', idx


class BalancedExemplarSet(ExemplarSet):
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, data, num_total_classes=100):
        super().__init__(buffer_size, device, data, num_total_classes)
        self.num_groups = 2
        self.per_cg_exemplar = {c: {g: [] for g in range(self.num_groups)} for c in range(num_total_classes)}
        self.per_cg_exemplar_features = {c: {g: [] for g in range(self.num_groups)} for c in range(num_total_classes)}

    def reduce_exemplar_set(self, num_new_classes):
        num_seen_classes = self.num_seen_classes + num_new_classes
        samples_per_cg = int(self.buffer_size / (num_seen_classes * self.num_groups))
        samples_per_class = round(self.buffer_size / num_seen_classes)

        for c in range(self.num_seen_classes):
            min_data = np.inf
            for g in range(len(self.per_cg_exemplar[c])):
                n_data = len(self.per_cg_exemplar[c][g])
                if n_data < min_data:
                    min_data = n_data
                    min_group = g
            if min_data < samples_per_cg:
                for g in range(len(self.per_cg_exemplar[c])):
                    n_data_to_keep = min_data if g == min_group else samples_per_class - min_data
                    self.per_cg_exemplar[c][g] = self.per_cg_exemplar[c][g][:n_data_to_keep]
                    self.per_cg_exemplar_features[c][g] = self.per_cg_exemplar_features[c][g][:n_data_to_keep]
            else:
                for g in range(len(self.per_cg_exemplar[c])):
                    self.per_cg_exemplar[c][g] = self.per_cg_exemplar[c][g][:samples_per_cg]
                    self.per_cg_exemplar_features[c][g] = self.per_cg_exemplar_features[c][g][:samples_per_cg]

        self.num_seen_classes = num_seen_classes

    def construct_exemplar_set(self, model, train_dataset):
        buffer_per_class = round(self.buffer_size / self.num_seen_classes)
        train_dataset = deepcopy(train_dataset)
        train_dataset.transform = self.fea_emb_transform
        num_classes = train_dataset.num_classes
        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, drop_last=False,
                                                   shuffle=False, sampler=None, **kwargs)
        # First, get class mean embedding
        model.eval()
        with torch.no_grad():
            if self.data == 'split_cifar_100s':
                class_means = torch.zeros(num_classes, 64).cuda()
            else:
                class_means = torch.zeros(num_classes, 512).cuda()

            per_class_n_data = torch.zeros(num_classes).cuda()
            per_class_images = {c: [] for c in range(num_classes)}
            per_class_groups = {c: [] for c in range(num_classes)}
            per_class_features = {c: [] for c in range(num_classes)}
            for samples in tqdm(train_loader):
                data, group, target, data_not_aug, _ = samples
                data, group, target = data.cuda(), group.cuda(), target.cuda()

                _, features, _ = model(data, get_inter=True)

                for c in range(num_classes):
                    if (target == c).sum() == 0:
                        continue
                    class_feature = features[target == c]
                    class_means[c] += class_feature.sum(0)
                    per_class_n_data[c] += (target == c).sum()
                    per_class_features[c].append(class_feature)

                    if self.data == 'split_cifar_100s':
                        per_class_images[c].append(data_not_aug[target == c])
                    else:
                        per_class_images[c].append(np.array(data_not_aug, dtype=object)[(target == c).cpu().numpy()])
                    per_class_groups[c].append(group[target == c].cpu())
                    # per_class_images[c].append(data_not_aug[target == c])

            for c in range(num_classes):
                per_class_features[c] = torch.cat(per_class_features[c])
                if self.data == 'split_cifar_100s':
                    per_class_images[c] = torch.cat(per_class_images[c])
                else:
                    per_class_images[c] = np.concatenate(per_class_images[c])
                per_class_groups[c] = torch.cat(per_class_groups[c])

            class_means /= per_class_n_data.unsqueeze(1)
            class_means = class_means / torch.norm(class_means, 2, 1).unsqueeze(1)
            class_offset = self.num_seen_classes - num_classes
            per_class_index_set = {c: [] for c in range(num_classes)}

            n_data_needed_per_gc = int(buffer_per_class / self.num_groups)
            num_train_data = train_dataset.num_data
            n_data_to_keep = np.full((self.num_groups, num_classes), n_data_needed_per_gc)
            n_data_to_keep[num_train_data < n_data_needed_per_gc] = num_train_data[num_train_data < n_data_needed_per_gc]
            n_data_to_keep[(num_train_data < n_data_needed_per_gc)[::-1, :]] = \
                buffer_per_class - n_data_to_keep[num_train_data < n_data_needed_per_gc]

            for c in range(num_classes):
                data_counter = torch.zeros(self.num_groups)
                for k in range(buffer_per_class):
                    if k == 0:
                        distance = torch.sqrt(torch.sum((class_means[c].unsqueeze(0) - per_class_features[c])**2, dim=1))
                    else:
                        feature_sum = per_class_features[c]
                        for g in range(self.num_groups):
                            if len(self.per_cg_exemplar_features[class_offset+c][g]) > 0:
                                feature_sum = feature_sum + torch.cat(self.per_cg_exemplar_features[class_offset+c][g]).sum(0).unsqueeze(0)

                        feature_sum = feature_sum / (k+1)
                        feature_sum = feature_sum / torch.norm(feature_sum, 2, 1).unsqueeze(1)
                        distance = torch.sqrt(torch.sum((class_means[c].unsqueeze(0) - feature_sum)**2, dim=1))
                    indicies = torch.argsort(distance)
                    for i in indicies:
                        if i.item() in per_class_index_set[c]:
                            continue
                        else:
                            g = per_class_groups[c][i].item()
                            if data_counter[g] < n_data_to_keep[g, c]:
                                per_class_index_set[c].append(i.item())
                        # i = torch.argmin(distance)
                                self.per_cg_exemplar[class_offset+c][g].append(per_class_images[c][i])
                                # self.per_cg_exemplar_groups[class_offset+c].append(per_class_groups[c][i].item())
                                self.per_cg_exemplar_features[class_offset+c][g].append(per_class_features[c][i])
                                data_counter[g] += 1
                                break

        self.examples = []
        self.labels = []
        self.groups = []

        for c in range(self.num_seen_classes):
            for g in range(self.num_groups):
                cg_images = self.per_cg_exemplar[c][g]
                self.examples.extend(cg_images)
                self.groups.extend([g] * len(cg_images))
                self.labels.extend([c] * len(cg_images))

        print('Buffer updated.')


def get_noise(noise_type):
    if noise_type == 'gaussian':
        return gaussian_noise
    elif noise_type == 'frost':
        return frost
    else:
        raise NotImplementedError


def get_transform(dataset, for_embedding=False):
    if dataset == 'split_cifar_100s':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])

        transform_list = [transforms.ToPILImage()]
        if not for_embedding:
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

    elif dataset.startswith('split_imagenet_100c'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        crop_size = 224
        if for_embedding:
            transform_list = [transforms.CenterCrop(crop_size)]
        else:
            transform_list = [transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip()]
    else:
        raise NotImplementedError

    transform_list.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform_list)
    return transform
