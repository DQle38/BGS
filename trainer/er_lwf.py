from __future__ import print_function

import torch
from tqdm import tqdm
import trainer.er

import torch.nn.functional as F


class Trainer(trainer.er.Trainer):
    def __init__(self, model, args, taskcla, num_total_data, buffer_size=None):
        super().__init__(model, args, taskcla, num_total_data, buffer_size)

        self.lamb = args.lamb
        self.T = args.temperature

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

            if buff_iter is not None and len(buff_loader) > 0:
                try:
                    buf_data, buf_target, buf_task_label, _ = next(buff_iter)
                except StopIteration:
                    buff_iter = iter(buff_loader)
                    buf_data, buf_target, buf_task_label, _ = next(buff_iter)

                buf_data, buf_target, buf_task_label = buf_data.cuda(), buf_target.cuda(), buf_task_label.cuda()
                data = torch.cat((data, buf_data))
                target = torch.cat((target, buf_target))
                task_label = torch.cat((task_label, buf_task_label))

            current_model_output = self.model(data)

            if self.args.continual == 'task':
                current_output = torch.stack(current_model_output)
                tmp = 0
                for t, i in self.taskcla:
                    tmp += current_output[t] * (task_label == t)[:, None]
                current_output = tmp

            elif self.args.continual == 'class':
                output = []
                for _t in range(t + 1):
                    output.append(current_model_output[_t])
                current_output = torch.cat(output, dim=1)
            else:
                current_output = current_model_output

            loss_ce = self.criterion(current_output, target)

            loss_distill = 0.
            if t > 0:
                with torch.no_grad():
                    prev_model_output = self.model_fixed(data)
                if self.args.continual in ['task', 'class']:
                    for old_t in range(t):
                        prev_model_old_output = prev_model_output[old_t]
                        current_model_old_output = current_model_output[old_t]

                        prev_p_old = F.softmax(prev_model_old_output / self.T, dim=1)
                        loss_distill += F.kl_div(torch.log_softmax(current_model_old_output / self.T, dim=1),
                                                 prev_p_old, reduction='batchmean')

                    loss_distill = loss_distill / t
                else:
                    prev_model_old_output = prev_model_output
                    current_model_old_output = current_model_output

                    prev_p_old = F.softmax(prev_model_old_output / self.T, dim=1)
                    loss_distill += F.kl_div(torch.log_softmax(current_model_old_output / self.T, dim=1),
                                             prev_p_old, reduction='batchmean')

            loss = loss_ce + self.lamb * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
