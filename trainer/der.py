from __future__ import print_function

import torch
from tqdm import tqdm
from trainer.er_util import Buffer
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, taskcla, num_total_data, buffer_size=None):
        super().__init__(model, args, taskcla)

        if buffer_size is not None:
            self.buffer_size = buffer_size
        else:
            buffer_size = int(num_total_data * args.buffer_ratio)
        self.buffer_size = buffer_size
        self.buffer = Buffer(buffer_size, torch.cuda.current_device(), args.dataset)

        self.lamb = args.lamb # hyperparameter for weighting the regularizaiton term

    def fill_buffer(self, train_dataset, t):
        train_loader, _ = self.get_dataloader(train_dataset, drop_last=False)
        num_data = 0.
        num_classes = train_dataset.num_classes
        num_total_tasks = self.args.start_task + self.args.n_tasks
        num_total_classes = num_classes * num_total_tasks

        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            if self.args.continual == 'class':
                target = target + int(num_classes * t)

            batch_size = data.shape[0]
            task_label = torch.full((batch_size,), t, dtype=torch.int)
            task_label = task_label.cuda()

            with torch.no_grad():
                data = data.cuda()
                if self.args.continual == 'class':
                    logit = []
                    model_output = self.model(data)
                    for _t in range(num_total_tasks):
                        logit.append(model_output[_t])
                    logit = torch.cat(logit, dim=1)

                else:
                    logit = self.model(data, task_id=task_label)

            self.buffer.add_data(examples=data_not_aug, labels=target, logits=logit,
                                 task_labels=task_label, groups=group,
                                 continual=self.args.continual, num_total_classes=num_total_classes)
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

    def _train_epoch(self, train_loader, optimizer, t, buff_loader, num_classes=10):
        self.model.train()
        num_total_tasks = self.args.start_task + self.args.n_tasks
        buff_iter = None
        if buff_loader is not None:
            buff_iter = iter(buff_loader)
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            data, target, = data.cuda(), target.cuda()

            batch_size = data.shape[0]
            task_label = torch.full((batch_size,), t, dtype=torch.int)
            task_label = task_label.cuda()

            if buff_iter is not None:
                try:
                    buf_data, buf_target, buf_logit, buf_task_label, buf_group = next(buff_iter)
                except StopIteration:
                    buff_iter = iter(buff_loader)
                    buf_data, buf_target, buf_logit, buf_task_label, buf_group = next(buff_iter)

                buf_data, buf_target, buf_task_label = buf_data.cuda(), buf_target.cuda(), buf_task_label.cuda()
                buf_logit = buf_logit.cuda()
                data = torch.cat((data, buf_data))
                target = torch.cat((target, buf_target))
                task_label = torch.cat((task_label, buf_task_label))

            if self.args.continual == 'task':
                output = self.model(data, task_id=task_label)
                logit = output
            elif self.args.continual == 'class':
                logit = []
                model_output = self.model(data)
                for _t in range(num_total_tasks):

                    logit.append(model_output[_t])
                output = model_output[t]
                logit = torch.cat(logit, dim=1)

            else:
                output = self.model(data)
                logit = output

            if buff_iter is not None:
                loss_ce = self.criterion(output[:batch_size], target[:batch_size])
                loss_buf = 0.
                if self.args.continual == 'class':
                    logit_diff = (buf_logit-logit[batch_size:])
                    for _t in range(t):
                        mask = buf_task_label >= _t
                        loss_buf = loss_buf + logit_diff[mask, _t*num_classes:(_t+1)*num_classes].pow(2).sum()
                else:
                    loss_buf = (buf_logit-logit[batch_size:]).pow(2).sum()
                loss = loss_ce + self.lamb * loss_buf

            else:
                loss_ce = self.criterion(output, target)
                loss = loss_ce 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
