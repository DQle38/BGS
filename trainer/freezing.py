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
            self.model.eval()
            self._freeze(t)

        self.t = t

        num_classes = train_dataset.num_classes
        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset)

        if not (self.args.continual == 'domain' and t > 0):
            for epoch in range(self.args.epochs):
                self._train_epoch(train_loader, optimizer, t, num_classes=num_classes)

                train_loss, train_acc, _, _ = self.evaluate(self.model, train_loader, t)
                statement = '| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                    epoch + 1, train_loss, 100 * train_acc)
                print(statement, end='')

                test_loss, test_acc, _, _ = self.evaluate(self.model, test_loader, t)

                statement = ' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc)
                print(statement, end='')

                print()
                scheduler.step()


    def _freeze(self, t):
        for name, param in self.model.named_parameters():
            if self.args.continual == 'task':
                if name.startswith('fc'):
                    head_number = int(name.split('.')[1])
                    if head_number < t:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

            elif self.args.continual == 'class':
                if not name.startswith('fc'):
                    param.requires_grad = False
            else:
                param.requires_grad = False
            print(name, param.requires_grad)

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10):
        self.model.train() if t == 0 else self.model.eval()

        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            data, target = data.cuda(), target.cuda()

            if self.args.continual == 'task':
                output = self.model(data)[t]
            elif self.args.continual == 'class':
                output = []
                model_output = self.model(data)
                for _t in range(t+1):
                    output.append(model_output[_t])
                output = torch.cat(output, dim=1)
                target = target + int(num_classes * t)

            else:
                output = self.model(data)

            loss = self.criterion(output, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


