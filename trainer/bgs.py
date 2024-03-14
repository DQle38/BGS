from __future__ import print_function

import torch
from tqdm import tqdm
from trainer.er_util import BufferGDumbBalanced
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, taskcla, num_total_data, buffer_size=None):
        super().__init__(model, args, taskcla)
        if buffer_size is not None:
            self.buffer_size = buffer_size
        else:
            buffer_size = int(num_total_data * args.buffer_ratio)

        self.buffer_size = buffer_size
        self.buffer = BufferGDumbBalanced(buffer_size, torch.cuda.current_device(), args.dataset)
        self.last_task = args.start_task + args.n_tasks - 1

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
        if self.args.modelpath is not None:
            self._fix_model()

        buff_loader, _ = self.get_dataloader(self.buffer, test_dataset[t], batch_size=self.args.bs)
        num_classes = train_dataset[t].num_classes

        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)
        for epoch in range(self.args.epochs):
            self._train_epoch(buff_loader, optimizer, t, num_classes=num_classes)
            scheduler.step()

    def _fix_model(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
            print(name, param.requires_grad)

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10):
        self.model.train()
        for samples in tqdm(train_loader):
            data, target, task_label, __ = samples
            data, target, task_label = data.cuda(), target.cuda(), task_label.cuda()

            inputs = data

            if self.args.continual == 'task':
                output = self.model(inputs, task_id=task_label)
            elif self.args.continual == 'class':
                target = target + (num_classes * task_label).long()

                output = []
                model_output = self.model(data)
                for _t in range(t + 1):
                    output.append(model_output[_t])
                output = torch.cat(output, dim=1)
            else:
                output = self.model(inputs)

            loss = self.criterion(output, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
