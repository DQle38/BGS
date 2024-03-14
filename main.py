import numpy as np

import torch
from torch.utils.data import DataLoader
from arguments import get_args
from utils import set_seed, check_dirs, load_datasets

import trainer
import utils
import networks
import wandb
wandb.login()


def do_continual_learning(trainer_, data_dict, args, save_dir, log_name):
    n_tasks = args.n_tasks if args.n_tasks is not None else len(data_dict['taskcla'])
    num_groups, taskcla = data_dict['num_groups'], data_dict['taskcla'],

    result_mat = utils.init_result_logs(taskcla, args.skew_ratio)
    kwargs = {'num_workers': args.n_workers, 'pin_memory': True}
    for t, ncla in data_dict['taskcla']:
        print("current task:", t)

        # train model on a new task data
        set_seed(args.seed, no_numpy=True)
        if args.start_task <= t < args.start_task + n_tasks:
            if 'gdumb' in args.trainer or 'bgs' == args.trainer or args.trainer == 'packnet_bgs':
                trainer_.train(data_dict['train_datasets'], data_dict['test_datasets'], t)
            else:
                trainer_.train(data_dict['train_datasets'][t], data_dict['test_datasets'][t], t)

        elif t >= args.start_task + n_tasks:
            break

        # method-wise processing after learning the task
        if args.trainer == 'ewc':
            trainer_.update_fisher(t, data_dict['train_datasets'][t])
        elif args.trainer == 'mas':
            trainer_.update_omega(t, data_dict['train_datasets'][t])
        elif args.trainer in ['er', 'der', 'eeil', 'ssil', 'groupdro_eeil']:
            trainer_.fill_buffer(data_dict['train_datasets'][t], t)
            if args.trainer == 'eeil':
                trainer_.balanced_fintuning(t)
        elif args.trainer == 'icarl':
            num_new_classes = taskcla[t][1]
            trainer_.buffer.reduce_exemplar_set(num_new_classes)
            trainer_.buffer.construct_exemplar_set(trainer_.model, data_dict['train_datasets'][t])
        elif args.trainer == 'packnet':
            trainer_.prune_n_retrain(data_dict['train_datasets'][t], data_dict['test_datasets'][t], t)

        # evaluate on learned tasks
        for u in range(t + 1):
            test_loader = DataLoader(data_dict['test_datasets'][u], args.bs, shuffle=False, **kwargs)
            skew_ratio = None if args.dataset == 'celeba' else data_dict['skew_ratios'][u]
            eval_results = trainer_.evaluate(trainer_.model, test_loader, u,
                                             skew_ratio=skew_ratio, cur_t=t)

            bmr = None
            if args.dataset in ['split_cifar_100s', 'split_imagenet_100c']:
                bmr = utils.get_bmr(trainer_, trainer_.model, data_dict['test_datasets'][u], u, args, t)
            result_mat = utils.update_result_logs(result_mat, eval_results, bmr, t, u)
            test_acc = result_mat['acc'][t, u]

            print('>>> Test on task {:2d}: acc={:5.1f}% <<<'.format(u, 100 * test_acc))

        utils.save_model_n_log(trainer_, result_mat, t, args.save_model, save_dir, log_name,
                               args.start_task, args.n_tasks)
    utils.print_continual_results(result_mat)


def main(args):
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    wandb.init(
            project='none',
            entity='none',
            name=args.ex_id,
            settings=wandb.Settings(start_method="fork")
    )

    set_seed(args.seed)
    if args.save_model:
        args, save_dir, log_name = check_dirs(args)
    else:
        save_dir, log_name = None, None

    data_dict = load_datasets(args)

    wandb.config.update(args)

    # Get the required model
    model = networks.ModelFactory.get_model(args, data_dict['taskcla']).cuda()
    trainer_ = trainer.TrainerFactory.get_trainer(model, args, data_dict)
    do_continual_learning(trainer_, data_dict, args, save_dir, log_name)

    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)
