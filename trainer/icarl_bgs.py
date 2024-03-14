from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import trainer.icarl
from copy import deepcopy
import torch.utils.data as data
from trainer.icarl_util import BalancedExemplarSet
from utils import get_accuracy



class Trainer(trainer.icarl.Trainer):
    def __init__(self, model, args, taskcla, num_total_data, buffer_size=None):
        super().__init__(model, args, taskcla, num_total_data, buffer_size)

        self.buffer = BalancedExemplarSet(self.buffer_size, torch.cuda.current_device(), args.dataset, self.total_num_classes)
        self.last_task = args.start_task + args.n_tasks -1

    def train(self, train_dataset, test_dataset, t=0):
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)

        num_classes = train_dataset.num_classes
        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset, batch_size=self.args.bs)
        train_dataset_to_concat = deepcopy(train_dataset)

        if t > 0:
            self._update_frozen_model()
            if len(self.buffer) > 0:
                train_dataset_to_concat.labels = train_dataset_to_concat.labels + int(num_classes * t)
                train_dataset_w_buffer = data.ConcatDataset([train_dataset_to_concat, self.buffer])
            else:
                train_dataset_w_buffer = train_dataset_to_concat
            train_loader_w_buffer, _ = self.get_dataloader(train_dataset_w_buffer, batch_size=self.args.bs)

        self.t = t

        for epoch in range(self.args.epochs):
            if t > 0:
                self._train_epoch(train_loader_w_buffer, optimizer, t, num_classes=num_classes)
            else:
                self._train_epoch(train_loader, optimizer, t, num_classes=num_classes)

            train_loss, train_acc, _, _ = self.evaluate(self.model, train_loader, t, for_training=True)
            statement = '| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                epoch + 1, train_loss, 100 * train_acc)
            print(statement, end='')

            test_loss, test_acc, _, _ = self.evaluate(self.model, test_loader, t, for_training=True)
            statement = ' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc)
            print(statement, end='')
            scheduler.step()

        self.buffer.reduce_exemplar_set(num_classes)
        self.buffer.construct_exemplar_set(self.model, train_dataset)
        if t == self.last_task:
            optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)
            buff_loader, _ = self.get_dataloader(self.buffer, batch_size=self.args.bs)

            for epoch in range(self.args.epochs):
                self._train_epoch_bgs(buff_loader, optimizer, t)
                scheduler.step()

        print('Training Finished!')

    def _train_epoch_bgs(self, train_loader, optimizer, t):
        self.model.train()
        for samples in tqdm(train_loader):

            data, _, target, _, _ = samples
            data, target, = data.cuda(), target.cuda()

            output = []
            model_output = self.model(data)
            for _t in range(t + 1):
                output.append(model_output[_t])
            output = torch.cat(output, dim=1)

            loss_ce = self.criterion(output, target)
            optimizer.zero_grad(set_to_none=True)
            loss_ce.backward()
            optimizer.step()

    def evaluate(self, model, loader, t=None, cur_t=None, skew_ratio=None, for_training=False):

        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        if hasattr(loader.dataset, 'skew_ratio'):
            skew_ratio = loader.dataset.skew_ratio if skew_ratio is None else skew_ratio

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_num = 0
            per_gc_hits = np.zeros((num_groups, num_classes))
            per_gc_num = np.zeros((num_groups, num_classes))

            if self.args.dataset == 'split_cifar_100s':
                exemplar_means = torch.zeros(self.buffer.num_seen_classes, 64).cuda()
            else:
                exemplar_means = torch.zeros(self.buffer.num_seen_classes, 512).cuda()

            for c in range(self.buffer.num_seen_classes):
                n_data = 0
                for g in range(self.buffer.num_groups):
                    exemplar_means[c] += torch.stack(self.buffer.per_cg_exemplar_features[c][g]).sum(0)
                    n_data += len(self.buffer.per_cg_exemplar_features[c][g])
                exemplar_means[c] = exemplar_means[c] / n_data
                exemplar_means[c] = exemplar_means[c] / torch.norm(exemplar_means[c], 2, 0)

            # Loop batches
            for samples in loader:
                data, group, target, _, _ = samples
                data, group, target = data.cuda(), group.cuda(), target.cuda()
                in_task_class_label = target
                target = target + int(num_classes * t)

                _, features, model_output = model(data, get_inter=True)

                output = []
                for _t in range(t + 1):
                    output.append(model_output[_t])
                output = torch.cat(output, dim=1)

                batch_size = len(target)
                if for_training:
                    preds = torch.argmax(output, 1)
                else:
                    means = torch.stack([exemplar_means] * batch_size)
                    means = means.transpose(1, 2).cuda()

                    features = features.unsqueeze(2)
                    features = features.expand_as(means)
                    dists = (features - means).pow(2).sum(1).squeeze()
                    _, preds = dists.min(1)

                loss = nn.CrossEntropyLoss(reduction='none')(output, target)
                total_loss += loss.sum().item()
                hits = (preds == target).float().squeeze()

                if hits.dim() == 0:
                    hits = hits.unsqueeze(0)

                for g in range(num_groups):
                    if (group == g).sum() > 0:
                        for c in range(num_classes):
                            if ((group == g) * (in_task_class_label == c)).sum() > 0:
                                per_gc_hits[g, c] += \
                                    hits[(group == g) * (in_task_class_label == c)].sum().data.cpu().numpy()
                                per_gc_num[g, c] += \
                                    ((group == g) * (in_task_class_label == c)).sum().data.cpu().numpy()

                total_num += data.shape[0]

            per_gc_num[per_gc_num == 0] += 1e-4
            total_loss /= total_num
            acc = get_accuracy(per_gc_hits, per_gc_num, self.args, loader.dataset, t, skew_ratio)

        return total_loss, acc, per_gc_hits, per_gc_num

