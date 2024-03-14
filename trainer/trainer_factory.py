import copy
import torch
import numpy as np
from importlib import import_module
from utils import get_accuracy


class TrainerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(myModel, args, data_dict=None, taskcla=None):
        taskcla = data_dict['taskcla'] if taskcla is None else taskcla
        trainer = import_module(f'trainer.{args.trainer}')
        rehearsal_based_methods = ['er', 'er_lwf', 'eeil', 'ssil', 'gdumb', 'der', 'bgs',
                                   'er_bgs', 'er_lwf_bgs', 'eeil_bgs', 'ssil_bgs', 'der_bgs',
                                   'groupdro_eeil', 'packnet_bgs', 'icarl', 'icarl_bgs']
        if args.trainer in rehearsal_based_methods:
            num_total_data = 0
            for dataset in data_dict['train_datasets']:
                num_total_data += len(dataset)
            num_total_data = num_total_data - len(data_dict['train_datasets'][-1])
            buffer_size = args.buffer_size
            return trainer.Trainer(myModel, args, taskcla, num_total_data, buffer_size=buffer_size)
        else:
            return trainer.Trainer(myModel, args, taskcla)


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this.
    '''
    def __init__(self, model, args, taskcla):
        self.model = model
        self.args = args
        self.taskcla = taskcla
        self.current_lr = args.lr
        self.ce = torch.nn.CrossEntropyLoss()

    def _update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def _get_optimzier_n_scheduler(self, optim_type, init_lr, model=None):
        model = model if model is not None else self.model
        if optim_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=self.args.decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=self.args.decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return optimizer, scheduler

    def _update_log(self, logger, **kwargs):
        for key, value in kwargs.items():
            logger[key].append(value)
        return logger

    def get_dataloader(self, train_dataset, test_dataset=None, get_balanced_loader=False,
                       batch_size=None, drop_last=True):

        kwargs = {'num_workers': self.args.n_workers, 'pin_memory': True}
        batch_size = batch_size if batch_size is not None else self.args.bs
        sampler = None
        shuffle = True
        if get_balanced_loader:
            from torch.utils.data.sampler import WeightedRandomSampler
            weights = train_dataset.make_weights()
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last,
                                                   shuffle=shuffle, sampler=sampler, **kwargs)
        test_loader = None
        if not test_dataset is None:
            test_loader = torch.utils.data.DataLoader(test_dataset, self.args.bs, drop_last=False,
                                                      shuffle=False, **kwargs)
        return train_loader, test_loader

    def evaluate(self, model, loader, t=None, cur_t=None, skew_ratio=None):
        return self._eval(model, loader, t, cur_t, skew_ratio)

    def _eval(self, model, loader, t=None, cur_t=None, skew_ratio=None):
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        if hasattr(loader.dataset, 'skew_ratio'):
            skew_ratio = loader.dataset.skew_ratio if skew_ratio is None else skew_ratio

        if self.args.continual == 'class' and cur_t is None:
            cur_t = t
            
        if self.args.trainer == 'packnet':
            weight_dict = self.pruner.apply_mask(model, t + 1)

        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_num = 0
            per_gc_hits = np.zeros((num_groups, num_classes))
            per_gc_num = np.zeros((num_groups, num_classes))

            # Loop batches
            for samples in loader:
                data, group, target, _, _ = samples
                data, group, target = data.cuda(), group.cuda(), target.cuda()
                in_task_class_label = target
                if self.args.continual == 'task':
                    output = model(data)[t]
                elif self.args.continual == 'class':
                    output = []
                    model_output = model(data)
                    for _t in range(cur_t + 1):
                        output.append(model_output[_t])
                    output = torch.cat(output, dim=1)
                    target = target + int(num_classes * t)
                else:
                    output = model(data)

                loss = torch.nn.CrossEntropyLoss(reduction='none')(output, target)
                total_loss += loss.sum().item()

                pred = torch.argmax(output, 1)
                hits = (pred == target).float().squeeze()

                if hits.dim() == 0:
                    hits = hits.unsqueeze(0)

                for g in range(num_groups):
                    if (group == g).sum() > 0:
                        for c in range(num_classes):
                            if ((group == g) * (in_task_class_label == c)).sum() > 0:
                                per_gc_hits[g, c] += \
                                    hits[(group == g) * (in_task_class_label == c)].sum().data.cpu().numpy()
                                per_gc_num[g, c] += \
                                    ((group == g) * (in_task_class_label == c)).sum().data.cpu().numpy()

                total_num += data.shape[0]

            per_gc_num[per_gc_num == 0] += 1e-4
            total_loss /= total_num
            acc = get_accuracy(per_gc_hits, per_gc_num, self.args, loader.dataset, t, skew_ratio)
        
        if self.args.trainer == 'packnet':
            self.pruner.restore_weight(model, weight_dict)

        return total_loss, acc, per_gc_hits, per_gc_num

    def criterion(self, output, targets):
        return self.ce(output, targets)
