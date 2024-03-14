from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from trainer.er_util import BufferGDumbBalanced
import trainer
from copy import deepcopy


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, taskcla, num_total_data, buffer_size=None):
        super().__init__(model, args, taskcla)

        if buffer_size is not None:
            self.buffer_size = buffer_size
        else:
            buffer_size = int(num_total_data * args.buffer_ratio)
        self.buffer_size = buffer_size
        self.buffer = BufferGDumbBalanced(buffer_size, torch.cuda.current_device(), args.dataset)
        self.ce = torch.nn.CrossEntropyLoss(reduction='sum')
        self.T = args.temperature
        self.last_task = args.start_task + args.n_tasks - 1

    def fill_buffer(self, train_dataset, t):
        self.buffer.add_new_classes(train_dataset.num_classes)

        train_loader, _ = self.get_dataloader(train_dataset, drop_last=False)
        num_data = 0.
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples

            batch_size = data.shape[0]
            task_label = torch.full((batch_size,), t, dtype=torch.int64)
            task_label = task_label.cuda()

            stop_flag = self.buffer.add_data(examples=data_not_aug[:batch_size], labels=target[:batch_size],
                                             task_labels=task_label[:batch_size], groups=group[:batch_size],
                                             continual=self.args.continual)
            if stop_flag:
                break
            num_data += batch_size

        print('Filled Buffer with task {} data: {}'.format(t, num_data))

    def _construct_training_set(self, train_dataset, t):
        train_dataset = DatasetWrapper(deepcopy(train_dataset), t)
        if t > 0:
            train_dataset = data.ConcatDataset([train_dataset, self.buffer])
        return train_dataset

    def balanced_fintuning(self, t):
        self._update_frozen_model()
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr/10)
        bft_epochs = int(self.args.epochs*3/4)
        buff_batch_size = min(self.args.bs, len(self.buffer))
        if buff_batch_size > 0:
            buff_loader, _ = self.get_dataloader(self.buffer, batch_size=buff_batch_size)

        for epoch in range(bft_epochs):
            self._train_epoch(buff_loader, optimizer, t)
            scheduler.step()

    def train(self, train_dataset, test_dataset, t=0):
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)

        if t > 0:
            self._update_frozen_model()

        self.t = t

        num_classes = train_dataset.num_classes
        train_dataset_w_buffer = self._construct_training_set(train_dataset, t)
        train_loader_w_buffer, test_loader = self.get_dataloader(train_dataset_w_buffer, test_dataset, batch_size=self.args.bs)
        train_loader, _ = self.get_dataloader(train_dataset, batch_size=self.args.bs)

        for epoch in range(self.args.epochs):
            self._train_epoch(train_loader_w_buffer, optimizer, t, num_classes=num_classes)
            
            train_loss, train_acc, _, _ = self.evaluate(self.model, train_loader, t)

            statement = '| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                epoch + 1, train_loss, 100 * train_acc)
            print(statement, end='')

            test_loss, test_acc, _, _= self.evaluate(self.model, test_loader, t)

            statement = ' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc)
            print(statement, end='')
            scheduler.step()

        self.fill_buffer(train_dataset, t)
        if t == self.last_task:
            self._fix_model()
            optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)
            buff_loader, _ = self.get_dataloader(self.buffer, batch_size=self.args.bs)
            for epoch in range(self.args.epochs):
                self._train_epoch_bgs(buff_loader, optimizer, t, num_classes=num_classes)
                scheduler.step()
        else:
            self.balanced_fintuning(t)


    def _train_epoch(self, train_loader, optimizer, t, num_classes=10, for_bft=False):
        self.model.train()
        for samples in tqdm(train_loader):
            data, target, task_label, _ = samples
            data, target, task_label = data.cuda(), target.cuda(), task_label.cuda()

            target = target + (num_classes * task_label).long()

            output = []
            model_output = self.model(data)
            for _t in range(t + 1):
                output.append(model_output[_t])
            output = torch.cat(output, dim=1)

            loss_distill = 0.
            if t > 0:
                with torch.no_grad():
                    prev_model_output = self.model_fixed(data)

                if for_bft:
                    prev_model_new_output = prev_model_output[t][task_label==t]
                    current_model_new_output = model_output[t][task_label==t]
                    prev_p_new = F.softmax(prev_model_new_output / self.T, dim=1)
                    loss_distill = F.kl_div(torch.log_softmax(current_model_new_output / self.T, dim=1),
                                             prev_p_new, reduction='batchmean') * (self.T ** 2)
                else:
                    for old_t in range(t):
                        prev_model_old_output = prev_model_output[old_t]
                        current_model_old_output = model_output[old_t]
                        prev_p_old = F.softmax(prev_model_old_output / self.T, dim=1)
                        loss_distill += F.kl_div(torch.log_softmax(current_model_old_output / self.T, dim=1),
                                                 prev_p_old, reduction='batchmean') * (self.T ** 2)

            loss_ce = self.criterion(output, target)
            optimizer.zero_grad()
            (loss_ce + loss_distill).backward()
            optimizer.step()

    def _fix_model(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
            print(name, param.requires_grad)

    def _train_epoch_bgs(self, train_loader, optimizer, t, num_classes=10):
        self.model.train()
        for samples in tqdm(train_loader):
            data, target, task_label, __ = samples
            data, target, task_label = data.cuda(), target.cuda(), task_label.cuda()

            if self.args.continual == 'task':
                output = self.model(data, task_id=task_label)

            elif self.args.continual == 'class':
                target = target + (num_classes * task_label).long()
                output = []
                model_output = self.model(data)
                for _t in range(t + 1):
                    output.append(model_output[_t])
                output = torch.cat(output, dim=1)
            else:
                output = self.model(data)

            loss_ce = self.criterion(output, target)
            optimizer.zero_grad(set_to_none=True)
            loss_ce.backward()
            optimizer.step()


class DatasetWrapper(data.Dataset):
    def __init__(self, dataset, task_id):
        super(DatasetWrapper, self).__init__()
        self.dataset = dataset
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        samples = self.dataset[index]
        data, group, target, _, _ = samples
        return data, torch.tensor(target), torch.tensor(self.task_id), torch.tensor(group, dtype=torch.float32)
