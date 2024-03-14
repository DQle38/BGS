from __future__ import print_function

import torch
import torch.nn.functional as F
from tqdm import tqdm

import trainer.vanilla


class Trainer(trainer.vanilla.Trainer):
    def __init__(self, model, args, taskcla):
        super().__init__(model, args, taskcla)
        self.lamb = args.lamb
        self.T = args.temperature

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10):
        self.model.train()
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            data, target = data.cuda(), target.cuda()

            current_model_output = self.model(data)

            if self.args.continual == 'task':
                current_output = current_model_output[t]
            elif self.args.continual == 'class':
                output = []
                for _t in range(t+1):
                    output.append(current_model_output[_t])
                current_output = torch.cat(output, dim=1)
                target = target + int(num_classes * t)

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

