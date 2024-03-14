from __future__ import print_function

import torch

import torch.utils.data as td
from tqdm import tqdm
import trainer.vanilla


class Trainer(trainer.vanilla.Trainer):
    def __init__(self, model, args, taskcla):
        super().__init__(model, args, taskcla)

        self.lamb = args.lamb

    def criterion(self, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if self.t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                     self.model_fixed.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.ce(output, targets) + self.lamb * loss_reg

    def fisher_matrix_diag(self, t):
        # Init
        fisher = {}
        for n, p in self.model.named_parameters():
            fisher[n] = 0 * p.data
        # Compute
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        num_data = 0
        for samples in tqdm(self.fisher_loader):
            data, _, target, _, _ = samples
            # data, _, target = samples
            data, target = data.cuda(), target.cuda()

            # Forward and backward
            self.model.zero_grad()
            outputs = self.model.forward(data)[t] if self.args.continual == 'task' else self.model(data)
            loss = criterion(outputs, target)
            loss.backward()
            num_data += len(target)
            # Get gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)*len(target)
        # Mean
        with torch.no_grad():
            for n, _ in self.model.named_parameters():
                fisher[n] = fisher[n] / num_data
        return fisher

    def update_fisher(self, t, train_dataset):
        kwargs = {'num_workers': self.args.n_workers, 'pin_memory': True}
        # self.t = t
        self.fisher_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, **kwargs)
        if t > 0:
            fisher_old = {}
            for n, _ in self.model.named_parameters():
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = self.fisher_matrix_diag(t)
        if t > 0:
            for n, _ in self.model.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (t + 1)
