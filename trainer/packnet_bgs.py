from __future__ import print_function

import torch
from tqdm import tqdm
from trainer.er_util import BufferGDumbBalanced
import trainer

from trainer.packnet import SparsePruner
from copy import deepcopy


class Trainer(trainer.packnet.Trainer):
    def __init__(self, model, args, taskcla, num_total_data, buffer_size=None):
        super().__init__(model, args, taskcla)
        if buffer_size is not None:
            self.buffer_size = buffer_size
        else:
            buffer_size = int(num_total_data * args.buffer_ratio)

        self.buffer_size = buffer_size
        self.buffer = BufferGDumbBalanced(buffer_size, torch.cuda.current_device(), args.dataset)
        self.last_task = args.start_task + args.n_tasks -1

        if self.args.modelpath is not None:
            mask_dict = torch.load(self.args.modelpath[:-3] + '_mask.pt')
            self.pruner.current_masks = mask_dict

    def fill_buffer(self, train_dataset, t):
        self.buffer.add_new_classes(train_dataset.num_classes)

        train_loader, _ = self.get_dataloader(train_dataset, drop_last=False)
        for samples in tqdm(train_loader):
            data, group, target, data_not_aug, _ = samples

            batch_size = data.shape[0]
            task_label = torch.full((batch_size,), t, dtype=torch.int)
            task_label = task_label.cuda()

            stop_flag = self.buffer.add_data(examples=data_not_aug, labels=target,
                                             task_labels=task_label, groups=group,
                                             continual=self.args.continual)
            if stop_flag:
                break

        print('Filled Buffer with task {} data'.format(t))

    def train(self, train_dataset, test_dataset, t=0, masks=None):
        self.fill_buffer(train_dataset[t], t)
        if t < self.last_task:
            return

        # copy models for each task
        self.model_dict = {}
        for _t in range(t+1):
            model = deepcopy(self.model)
            self.pruner.apply_mask(model, _t+1)
            self.model_dict[_t] = model.eval()

        self.t = t
        if self.args.modelpath is not None:
            self._fix_model()

        optimizer, scheduler = self._get_optimzier_n_scheduler(self.args.optim, self.args.lr)
        buff_loader, _ = self.get_dataloader(self.buffer, test_dataset[t], batch_size=self.args.bs)
        num_classes = train_dataset[t].num_classes
        for epoch in range(self.args.epochs):
            self._train_epoch(buff_loader, optimizer, t, num_classes=num_classes)
            scheduler.step()

    def _fix_model(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
            print(name, param.requires_grad)

    def _train_epoch(self, train_loader, optimizer, t, num_classes=10):
        self.model.train()

        for samples in tqdm(train_loader):
            data, target, task_label, __ = samples
            data, target, task_label = data.cuda(), target.cuda(), task_label.cuda()
            inputs = data

            feature = torch.empty((data.shape[0], 64), dtype=torch.float32).cuda()
            with torch.no_grad():
                for _t in range(t + 1):
                    if (task_label == _t).sum() == 0:
                        continue
                    _, _feature, _ = self.model_dict[_t].forward(inputs, get_inter=True)
                    mask = (task_label == _t).nonzero(as_tuple=True)[0]
                    feature[mask] = _feature[mask]

            output = self.model.forward_head(feature, task_id=task_label)
            loss = self.criterion(output, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


