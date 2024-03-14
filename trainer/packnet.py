from __future__ import print_function


from tqdm import tqdm

import trainer
import torch
import torch.nn as nn

from copy import deepcopy


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, taskcla):
        super().__init__(model, args, taskcla)
        self.masks_dict = {}

        """Init masks"""
        masks = {}
        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            if self.args.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                masks[module_idx] = mask

        self.pruner = SparsePruner(
                self.model, self.args.prune_ratio, masks,
                train_bias=False, train_bn=False)

    def train(self, train_dataset, test_dataset, t=0):
        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)

        # Now, you can update self.t
        self.t = t

        num_classes = train_dataset.num_classes
        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset)

        if self.t > 0 :
            self.pruner.make_finetuning_mask()
            self.pruner.train_bias = False
            self.pruner.train_bn = False
        else:
            self.pruner.train_bias = True            
            self.pruner.train_bn = True

        # full training
        for epoch in range(self.args.epochs):
            self._train_epoch(train_loader, optimizer, t, num_classes=num_classes)
            self._evaluate(epoch, train_loader, test_loader, t, num_classes)
            scheduler.step()


    def _evaluate(self, epoch, train_loader, test_loader, t, num_classes=10):
        train_loss, train_acc, _, _ = self.evaluate(self.model, train_loader, t)

        statement = '| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(
            epoch + 1, train_loss, 100 * train_acc)
        print(statement, end='')

        test_loss, test_acc, _, _ = self.evaluate(self.model, test_loader, t)
        statement = ' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc)
        print(statement, end='')


    def prune_n_retrain(self, train_dataset, test_dataset, t):
        num_classes =train_dataset.num_classes
        train_loader, test_loader = self.get_dataloader(train_dataset, test_dataset)
        # Pruning
        self.prune()

        # Do final finetuning to improve results on pruned network.
        if self.args.post_prune_epochs:
            print('Doing some extra finetuning...')
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr/10, weight_decay=self.args.decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.post_prune_epochs)

            if t > 0:
                self.pruner.train_bias = False
                self.pruner.train_bn = False
            else:
                self.pruner.train_bias = True            
                self.pruner.train_bn = True   
            
            for epoch in range(self.args.post_prune_epochs):
                self._train_epoch(train_loader, optimizer, t)
                self._evaluate(epoch, train_loader, test_loader, t, num_classes)
            scheduler.step()

        print('-' * 16)
        print('Pruning summary:')
        print('-' * 16)

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10):
        if t > 0:
            self.model.train_nobn()
            # self.model.eval()
        else:
            self.model.train()
        param_dict = self._get_params()

        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples
            data, target = data.cuda(), target.cuda()

            output = self.model(data)[t] if self.args.continual == 'task' else self.model(data)
            loss = self.criterion(output, target)

            optimizer.zero_grad()
            loss.backward()

            # Set fixed param grads to 0.
            self.pruner.make_grads_zero()

            # Update params.
            optimizer.step()

            # recover weights for fixed filters
            self._restore_params(param_dict)
            # Set pruned weights to 0.
            self.pruner.make_pruned_zero()

    def prune(self):
        """Perform pruning."""
        print('Start pruning')
        masks = self.pruner.prune()
        self.check(True)

    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * float(num_zero) / num_params))

    def _get_params(self):
        param_dict = {}
        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            if self.args.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or 'BatchNorm' in str(type(module)):
                param_dict[module_idx] = {}
                weight = module.weight.data
                param_dict[module_idx]['weight'] = deepcopy(weight)
                if module.bias is not None:
                    bias = module.bias.data
                    param_dict[module_idx]['bias'] = deepcopy(bias)

        return param_dict

    def _restore_params(self, param_dict):
        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            if self.args.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.pruner.current_masks[module_idx]
                weight = param_dict[module_idx]['weight']
                module.weight.data[layer_mask.ne(self.pruner.current_dataset_idx)] = \
                    weight[layer_mask.ne(self.pruner.current_dataset_idx)]
                # module.weight.data[layer_mask.ne(self.pruner.current_dataset_idx)] = 100.0
                if not self.pruner.train_bias:
                    # Biases are fixed.
                    if module.bias is not None:
                        bias = param_dict[module_idx]['bias']
                        module.bias.data.copy_(bias)

            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.pruner.train_bn:
                    weight = param_dict[module_idx]['weight']
                    bias = param_dict[module_idx]['bias']
                    module.weight.data.copy_(weight)
                    module.bias.data.copy_(bias)


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, train_bias, train_bn, continual='task'):
        self.model = model
        self.prune_perc = prune_perc
        self.train_bias = train_bias
        self.train_bn = train_bn
        self.continual = continual

        self.current_masks = previous_masks

        valid_key = list(self.current_masks.keys())[0]
        self.current_dataset_idx = self.current_masks[valid_key].max()        

    def pruning_mask(self, weights, previous_mask, layer_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(self.current_dataset_idx)]
        abs_tensor = tensor.abs()

        if len(self.prune_perc) == 1:
            prune_perc = self.prune_perc[0]
        else:
            prune_perc = self.prune_perc[self.current_dataset_idx - 1]

        cutoff_rank = round(prune_perc * tensor.numel())
        if cutoff_rank > 0:

            cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]

            # Remove those weights which are below cutoff and belong to current
            # dataset that we are training for.
            remove_mask = weights.abs().le(cutoff_value) * \
                previous_mask.eq(self.current_dataset_idx)

            # mask = 1 - remove_mask
            previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * float(mask.eq(0).sum()) / tensor.numel(), weights.numel()))
        return mask

    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """

        if len(self.prune_perc) == 1:
            prune_perc = self.prune_perc[0]
        else:
            prune_perc = self.prune_perc[self.current_dataset_idx - 1]

        print('Pruning each layer by removing %.2f%% of values' %
              (100 * prune_perc))
        
        valid_key = list(self.current_masks.keys())[0]        
        self.current_dataset_idx = self.current_masks[valid_key].max()        
        print(self.current_dataset_idx)

        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            if self.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask(
                    module.weight.data, self.current_masks[module_idx], module_idx)
                self.current_masks[module_idx] = mask.cuda()
                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks

        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            if self.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(
                        self.current_dataset_idx)] = 0.
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0.)

            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks

        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            if self.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def apply_mask(self, model, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
        weight_dict = {}
        for module_idx, (name, module) in enumerate(model.named_modules()):
            if self.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.current_masks[module_idx].cuda()
                weight_dict[module_idx] = deepcopy(weight)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0

        return weight_dict


    def restore_weight(self, model, weight_dict):
        for module_idx, (name, module) in enumerate(model.named_modules()):
            if self.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data.copy_(weight_dict[module_idx])


    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        self.current_dataset_idx += 1
        print('current task id ', self.current_dataset_idx)
        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            if self.continual == 'task' and 'fc' in name:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.current_masks[module_idx]
                mask[mask.eq(0)] = self.current_dataset_idx
