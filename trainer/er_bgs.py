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
            print(num_total_data)
            buffer_size = int(num_total_data * args.buffer_ratio)
        self.buffer_size = buffer_size
        self.last_task = args.start_task + args.n_tasks - 1
        self.buffer = BufferGDumbBalanced(buffer_size, torch.cuda.current_device(), args.dataset)

    def fill_buffer(self, train_dataset, t):
        self.buffer.add_new_classes(train_dataset.num_classes)

        train_loader, _ = self.get_dataloader(train_dataset, drop_last=False)
        num_data = 0.
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
            num_data += batch_size

        print('Filled Buffer with task {} data: {}'.format(t, num_data))

    def train(self, train_dataset, test_dataset, t=0):
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)

        buff_loader = None
        if t > 0:
            self._update_frozen_model()
            buff_batch_size = min(self.args.bs, len(self.buffer))
            if buff_batch_size > 0:
                buff_loader, _ = self.get_dataloader(self.buffer, batch_size=buff_batch_size)

        # Now, you can update self.t
        self.t = t

        num_classes = train_dataset.num_classes
        cur_batch_size = self.args.bs
        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset, batch_size=cur_batch_size)

        for epoch in range(self.args.epochs):
            self._train_epoch(train_loader, optimizer, t, buff_loader, num_classes=num_classes)
            
            train_loss, train_acc, _, _ = self.evaluate(self.model, train_loader, t)

            statement = '| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                epoch + 1, train_loss, 100 * train_acc)
            print(statement, end='')

            test_loss, test_acc, _, _ = self.evaluate(self.model, test_loader, t)
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


    def _train_epoch(self, train_loader, optimizer, t, buff_loader, num_classes=10):
        self.model.train()
        buff_iter = None
        if buff_loader is not None:
            buff_iter = iter(buff_loader)
        for samples in tqdm(train_loader):
            data, _, target, data_not_aug, _ = samples
            data, target, = data.cuda(), target.cuda()

            if self.args.continual == 'class':
                target = target + int(num_classes * t)

            batch_size = data.shape[0]
            task_label = torch.full((batch_size,), t, dtype=torch.int)
            task_label = task_label.cuda()

            if buff_iter is not None:
                try:
                    buf_data, buf_target, buf_task_label, _ = next(buff_iter)
                except StopIteration:
                    buff_iter = iter(buff_loader)
                    buf_data, buf_target, buf_task_label, _ = next(buff_iter)

                buf_data, buf_target, buf_task_label = buf_data.cuda(), buf_target.cuda(), buf_task_label.cuda()
                if self.args.continual == 'class':
                    buf_target = buf_target + (num_classes * buf_task_label).long()

                data = torch.cat((data, buf_data))
                target = torch.cat((target, buf_target))
                task_label = torch.cat((task_label, buf_task_label))

            if self.args.continual == 'task':
                output = self.model(data, task_id=task_label)
            elif self.args.continual == 'class':
                output = []
                model_output = self.model(data)
                for _t in range(t+1):
                    output.append(model_output[_t])
                output = torch.cat(output, dim=1)

            else:
                output = self.model(data)
            loss_ce = self.criterion(output, target)
            optimizer.zero_grad()
            loss_ce.backward()
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
