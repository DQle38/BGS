from __future__ import print_function
from tqdm import tqdm
import torch
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, taskcla):
        super().__init__(model, args, taskcla)

    def train(self, train_dataset, test_dataset, t=0):
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)

        if t > 0:
            self._update_frozen_model()

        self.t = t

        num_classes = train_dataset.num_classes
        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset)

        for epoch in range(self.args.epochs):
            self._train_epoch(train_loader, optimizer, t, num_classes=num_classes)
            train_loss, train_acc, _, _ = self.evaluate(self.model, train_loader, t)
            statement = '| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                epoch + 1, train_loss, 100 * train_acc)
            print(statement, end='')

            test_loss, test_acc, _, _ = self.evaluate(self.model, test_loader, t)
            statement = ' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc)
            print(statement, end='')

            scheduler.step()

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10):
        self.model.train()
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            data, target = data.cuda(), target.cuda()

            if self.args.continual == 'task':
                output = self.model(data)[t]
            elif self.args.continual == 'class':
                output = []
                model_output = self.model(data)
                target = target + int(num_classes * t)
                for _t in range(t+1):
                    output.append(model_output[_t])
                output = torch.cat(output, dim=1)
            else:
                output = self.model(data)

            loss = self.criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


