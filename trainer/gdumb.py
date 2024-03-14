from __future__ import print_function

import torch
import torch.optim as optim
from tqdm import tqdm
from trainer.er_util import BufferGDumb
import trainer
import numpy as np


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, taskcla, num_total_data, buffer_size=None):
        super().__init__(model, args, taskcla)
        if buffer_size is not None:
            self.buffer_size = buffer_size
        else:
            print(num_total_data)
            buffer_size = int(num_total_data * args.buffer_ratio)
        self.buffer_size = buffer_size
        self.buffer = BufferGDumb(buffer_size, torch.cuda.current_device(), args.dataset)
        self.last_task = args.start_task + args.n_tasks - 1
        self.cutmix_prob = args.cutmix_prob
        self.beta = args.beta
        self.use_cutmix = args.use_cutmix
        self.max_lr = 0.05
        self.min_lr = 0.0005

    def fill_buffer(self, train_dataset, t):
        self.buffer.add_new_classes(train_dataset.num_classes)

        train_loader, _ = self.get_dataloader(train_dataset, drop_last=False)
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            batch_size = data.shape[0]
            task_label = torch.full((batch_size,), t, dtype=torch.int)
            task_label = task_label.cuda()

            stop_flag = self.buffer.add_data(examples=data_not_aug, labels=target,
                                             task_labels=task_label, groups=group,
                                             continual=self.args.continual)
            if stop_flag:
                break

        print('Filled Buffer with task {} data'.format(t))

    def train(self, train_dataset, test_dataset, t=0):
        self.fill_buffer(train_dataset[t], t)
        if t < self.last_task:
            return

        self.t = t

        optimizer = optim.SGD(self.model.parameters(), lr=self.max_lr, momentum=0.9, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=self.min_lr)

        buff_loader, _ = self.get_dataloader(self.buffer, test_dataset[t], batch_size=16)
        num_classes = train_dataset[t].num_classes
        for epoch in range(self.args.epochs):

            # Handle lr scheduling
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.max_lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.max_lr
            else:  # and go!
                scheduler.step()

            self._train_epoch(buff_loader, optimizer, t, num_classes=num_classes)
            

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10):
        self.model.train()
        for samples in tqdm(train_loader):
            data, target, task_label, __ = samples
            data, target, task_label = data.cuda(), target.cuda(), task_label.cuda()

            r = np.random.rand(1)
            if self.beta > 0 and r < self.cutmix_prob and self.use_cutmix:
                if self.args.continual == 'class':
                    target = target + (num_classes * task_label).long()

                if self.args.continual == 'task':
                    loss = 0
                    for _t in task_label.unique():
                        task_mask = _t == task_label
                        task_inputs = data[task_mask]
                        task_target = target[task_mask]
                        # generate mixed sample
                        lam = np.random.beta(self.beta, self.beta)
                        rand_index = torch.randperm(task_inputs.size()[0]).cuda()
                        target_a = task_target
                        target_b = task_target[rand_index]
                        bbx1, bby1, bbx2, bby2 = self.rand_bbox(task_inputs.size(), lam)
                        task_inputs[:, :, bbx1:bbx2, bby1:bby2] = task_inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (task_inputs.size()[-1] * task_inputs.size()[-2]))

                        output = self.model(task_inputs)[int(task_label.item())]

                        loss = loss + self.criterion(output, target_a) * lam + self.criterion(output, target_b) * (1. - lam)

                else:
                    # generate mixed sample
                    lam = np.random.beta(self.beta, self.beta)
                    rand_index = torch.randperm(data.size()[0]).cuda()
                    target_a = target
                    target_b = target[rand_index]
                    bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
                    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
                    # compute output

                    if self.args.continual == 'class':
                        output = []
                        model_output = self.model(data)
                        for _t in range(t + 1):
                            output.append(model_output[_t])
                        output = torch.cat(output, dim=1)
                    else:
                        output = self.model(data)
                    loss = self.criterion(output, target_a) * lam + self.criterion(output, target_b) * (1. - lam)
            else:
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

                loss = self.criterion(output, target)
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            loss.backward()
            optimizer.step()

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2