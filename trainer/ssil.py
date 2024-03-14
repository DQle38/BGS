from __future__ import print_function

import torch
import torch.nn.functional as F
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
        self.ce = torch.nn.CrossEntropyLoss(reduction='sum')
        self.T = args.temperature

    def fill_buffer(self, train_dataset, t):
        train_loader, _ = self.get_dataloader(train_dataset, drop_last=False)
        num_data = 0.
        num_classes = train_dataset.num_classes
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            if self.args.continual == 'class':
                target = target + int(num_classes * t)

            batch_size = data.shape[0]
            task_label = torch.full((batch_size,), t, dtype=torch.int)
            task_label = task_label.cuda()

            self.buffer.add_data(examples=data_not_aug, labels=target,
                                 task_labels=task_label, groups=group)
            num_data += batch_size

        print('Filled Buffer with task {} data: {}'.format(t, num_data))

    def train(self, train_dataset, test_dataset, t=0):
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)

        buff_loader = None
        if t > 0:
            self._update_frozen_model()
            buff_batch_size = min(int(self.args.bs / 4), len(self.buffer))
            if buff_batch_size > 0:
                buff_loader, _ = self.get_dataloader(self.buffer, batch_size=buff_batch_size)

        self.t = t

        num_classes = train_dataset.num_classes
        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset, batch_size=self.args.bs)

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
        buff_iter = None
        if buff_loader is not None:
            buff_iter = iter(buff_loader)
        for samples in tqdm(train_loader):
            data, _, target, data_not_aug, _ = samples
            data, target, = data.cuda(), target.cuda()

            batch_size = data.shape[0]

            if buff_iter is not None and len(buff_loader) > 0:
                try:
                    buf_data, buf_target, buf_task_label, _ = next(buff_iter)
                except StopIteration:
                    buff_iter = iter(buff_loader)
                    buf_data, buf_target, buf_task_label, _ = next(buff_iter)

                buf_data, buf_target, buf_task_label = buf_data.cuda(), buf_target.cuda(), buf_task_label.cuda()
                data = torch.cat((data, buf_data))
                buf_batch_size = buf_data.shape[0]

            prev_output = []
            cur_output = []
            model_output = self.model(data)
            for _t in range(t+1):
                if _t == t:
                    cur_output.append(model_output[_t])
                else:
                    prev_output.append(model_output[_t])

            cur_output = torch.cat(cur_output, dim=1)[:batch_size]
            loss_ce_curr = self.ce(cur_output, target)

            loss_distill = 0.
            if t > 0:
                prev_output = torch.cat(prev_output, dim=1)[batch_size:]
                loss_ce_prev = self.ce(prev_output, buf_target)
                loss_ce = (loss_ce_curr + loss_ce_prev) / (batch_size + buf_batch_size)

                with torch.no_grad():
                    prev_model_output = self.model_fixed(data)

                for old_t in range(t):
                    prev_model_old_output = prev_model_output[old_t]
                    current_model_old_output = model_output[old_t]
                    prev_p_old = F.softmax(prev_model_old_output / self.T, dim=1)
                    loss_distill += F.kl_div(torch.log_softmax(current_model_old_output / self.T, dim=1),
                                             prev_p_old, reduction='batchmean') * (self.T ** 2)

            else:
                loss_ce = loss_ce_curr / batch_size

            optimizer.zero_grad()
            (loss_ce + loss_distill).backward()
            optimizer.step()
