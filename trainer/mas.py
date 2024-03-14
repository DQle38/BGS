from __future__ import print_function

import torch

import torch.utils.data as td
from tqdm import tqdm
import trainer.vanilla



class Trainer(trainer.vanilla.Trainer):
    def __init__(self, model, args, taskcla):
        super().__init__(model, args, taskcla)

        self.lamb = args.lamb
        self.omega = {}
        for n, _ in self.model.named_parameters():
            self.omega[n] = 0

    def criterion(self, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if self.t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                     self.model_fixed.named_parameters()):
                loss_reg += torch.sum(self.omega[name] * (param_old - param).pow(2)) / 2
        return self.ce(output, targets) + self.lamb * loss_reg

    def update_omega(self, t, train_dataset):
        kwargs = {'num_workers': self.args.n_workers, 'pin_memory': True}
        self.omega_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, **kwargs)
        # Compute
        self.model.train()
        for samples in tqdm(self.omega_loader):
            data, group, target, data_not_aug, _ = samples
            data, target = data.cuda(), target.cuda()
            # Forward and backward
            self.model.zero_grad()
            outputs = self.model.forward(data)[t] if self.args.continual == 'task' else self.model.forward(data)

            # Sum of L2 norm of output scores
            loss = torch.sum(outputs.norm(2, dim=-1))
            loss.backward()

            # Get gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.omega[n] += p.grad.data.abs() / len(self.omega_loader)

        return