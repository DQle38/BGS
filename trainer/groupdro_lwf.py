from __future__ import print_function
import torch
from tqdm import tqdm
import trainer.groupdro
import torch.nn.functional as F


class Trainer(trainer.groupdro.Trainer):
    def __init__(self, model, args, taskcla):
        super().__init__(model, args, taskcla)
        self.lamb = args.lamb
        self.T = args.temperature        

    def _train_epoch(self, train_loader, optimizer, t, num_classes, num_groups):
        self.model.train()
        num_subgroups = num_classes * num_groups
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, indexes = samples
            data, target, group = data.cuda(), target.cuda(), group.cuda()

            output = self.model(data)
            logits = output[t] if self.args.continual == 'task' else output

            loss = self.gdro_criterion(logits, target)

            subgroups = group * num_classes + target
            
            # calculate the groupwise losses
            group_map = (subgroups == torch.arange(num_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count == 0).float()  # avoid nans
            group_loss = (group_map @ loss.view(-1)) / group_denom

            # update q
            self.adv_probs = self.adv_probs * torch.exp(self.gamma * group_loss.data)
            self.adv_probs = self.adv_probs / (self.adv_probs.sum())  # proj

            robust_loss = group_loss @ self.adv_probs
            
            distill_loss = 0.
            if t > 0:
                with torch.no_grad():
                    prev_model_output = self.model_fixed(data)

                if self.args.continual == 'task':
                    for old_t in range(t):
                        prev_model_old_output = prev_model_output[old_t]
                        current_model_old_output = output[old_t]

                        prev_p_old = F.softmax(prev_model_old_output / self.T, dim=1)
                        distill_loss += F.kl_div(torch.log_softmax(current_model_old_output / self.T, dim=1),
                                                 prev_p_old, reduction='batchmean')
                    distill_loss = distill_loss / t

            optimizer.zero_grad()
            (robust_loss + self.lamb * distill_loss).backward()
            optimizer.step() 
