from __future__ import print_function

import torch

from tqdm import tqdm
import trainer.vanilla


class Trainer(trainer.vanilla.Trainer):
    def __init__(self, model, args, taskcla):
        super().__init__(model, args, taskcla)

        self.gamma = args.gamma
        self.gdro_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def train(self, train_dataset, test_dataset, t=0):
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)
        # Do not update self.t
        if t > 0:
            self._update_frozen_model()

        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset, get_balanced_loader=True)

        num_classes = train_dataset.num_classes
        num_groups = train_dataset.num_groups
        
        self.adv_probs = torch.ones(num_groups*num_classes).cuda() / (num_classes*num_groups)
        self.t = t

        for epoch in range(self.args.epochs):
            self._train_epoch(train_loader, optimizer, t, num_classes, num_groups)
            train_loss, train_acc, _, _ = self.evaluate(self.model, train_loader, t)
            statement = '| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                epoch + 1, train_loss, 100 * train_acc)

            print(statement, end='')

            test_loss, test_acc, _, _ = self.evaluate(self.model, test_loader, t)
            statement = ' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc)

            print(statement, end='')
            scheduler.step()
            

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10, num_groups=2):
        self.model.train()
        num_subgroups = num_classes * num_groups
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            data, group, target = data.cuda(), group.cuda(), target.cuda()

            output = self.model(data)
            logits = output[t] if self.args.continual == 'task' else output
            
            loss = self.gdro_criterion(logits, target)
            
            subgroups = group * num_classes + target
            # calculate the groupwise losses
            group_map = (subgroups == torch.arange(num_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ loss.view(-1))/group_denom

            # update q
            self.adv_probs = self.adv_probs * torch.exp(self.gamma*group_loss.data)
            self.adv_probs = self.adv_probs/(self.adv_probs.sum()) # proj
            
            robust_loss = group_loss @ self.adv_probs
            
            optimizer.zero_grad()
            robust_loss.backward()
            optimizer.step()    

