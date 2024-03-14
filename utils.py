import numpy as np
import os
import random
import torch
from typing import List
import data_handler
import importlib.util
import sys
import wandb


def set_seed(seed, no_numpy=False):
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if not no_numpy:
        np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_dirs(args):
    log_name = make_log_name(args)
    dataset = args.dataset
    if args.dataset in ['split_cifar_100s, split_imagenet_100c']:
        dataset += '_skew_random' if args.skew_ratio is None else f'_skew{args.skew_ratio}'
    elif 'celeba_two' in args.dataset:
        dataset += f'{args.skew_ratio}'
    elif 'split_imagenet_100c' in args.dataset:
        dataset += f'_{args.noise_type}'

    continual = args.continual
    save_dir = os.path.join('./trained_models', args.ex_id, dataset, continual, args.trainer)

    def make_folder(path):
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
        except OSError:
            print("Failed to create directory!!")

    make_folder(save_dir)
    return args, save_dir, log_name


def load_datasets(args):
    print('Load data...')

    kwargs = {'name': args.dataset,
              'seed': args.seed,
              }

    if args.skew_ratio is not None:
        kwargs['skew_ratio'] = args.skew_ratio

    if args.dataset == 'split_cifar_100s':
        kwargs['accumulation'] = args.accumulation
        kwargs['n_biased'] = args.n_biased
        kwargs['for_backward'] = args.for_backward

    if 'split_imagenet_100c' in args.dataset:
        kwargs['noise_type'] = args.noise_type

    tmp = data_handler.DatasetFactory.get_continual_datasets(n_tasks=args.n_tasks, start_task=args.start_task,
                                                             task_ids=args.task_seq, **kwargs)
    taskcla, num_groups, train_datasets, test_datasets, skew_ratios = tmp
    num_total_data = train_datasets[0].num_total_data
    data_dict = {'train_datasets': train_datasets, 'test_datasets': test_datasets, 'taskcla': taskcla,
                 'num_groups': num_groups, 'num_total_data': num_total_data, 'skew_ratios': skew_ratios}
    print('Num total data : ', num_total_data)

    return data_dict


def init_result_logs(taskcla, skew_ratio=None):

    total_tasks = len(taskcla)
    print( f'# of tasks : {total_tasks}')
    result_mat = {'acc': np.zeros((total_tasks, total_tasks), dtype=np.float32),
                  'dca': np.zeros((total_tasks, total_tasks), dtype=np.float32),
                  'skew_ratio': skew_ratio,
                  'bmr': np.zeros((total_tasks, total_tasks), dtype=np.float32),
                  }
    return result_mat


def update_result_logs(result_mat, eval_results, bmr=None, t=None, u=None):
    test_loss, test_acc, per_gc_hits, per_gc_num = eval_results
    per_gc_acc = per_gc_hits / per_gc_num
    classwise_difference = np.max(per_gc_acc, axis=0) - np.min(per_gc_acc, axis=0)
    dca = np.mean(classwise_difference)
    result_mat['acc'][t, u] = test_acc
    result_mat['dca'][t, u] = dca
    result_mat['bmr'][t, u] = bmr

    return result_mat


def print_continual_results(result_mat):
    print('*' * 100)
    print('Accuracies =')
    for i in range(result_mat['acc'].shape[0]):
        print('\t', end='')
        for j in range(result_mat['acc'].shape[1]):
            print('{:5.1f}% '.format(100 * result_mat['acc'][i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')


def save_model_n_log(trainer, result_mat, t=None, save_model=False, save_dir='', log_name='', start_task=0, n_tasks=10):
    def array2table(arr):
        if type(arr) != np.ndarray:
            return arr
        else:
            return list(arr)
    print('Save!')
    key = f'eval@task{t}'
    metrics = {}
    metrics[key] = {
        'acc': array2table(result_mat['acc']),
        'dca': array2table(result_mat['dca']),
        'bmr': array2table(result_mat['bmr']),
    }

    wandb.run.summary.update(metrics)
    if save_model:
        if hasattr(trainer, 'pruner'):
            torch.save(trainer.pruner.current_masks,
                       os.path.join(save_dir, log_name) + '_task{}_st{}_nt{}_mask.pt'.format(t, start_task, n_tasks))

        torch.save(trainer.model.state_dict(), os.path.join(save_dir, log_name) + '_task{}_st{}_nt{}.pt'.format(
            t, start_task, n_tasks)
        )


def get_accuracy(per_gc_hits, per_gc_num, args, dataset, t, skew_ratio=None):
    if args.dataset == 'celeba_two':
        assert skew_ratio is not None
        per_gc_weight = np.array([[1 - skew_ratio, skew_ratio],
                                  [skew_ratio, 1 - skew_ratio]])
        per_gc_weight = per_gc_weight * 0.5
        acc = (per_gc_hits / per_gc_num) * per_gc_weight
        acc = acc.sum()

    elif args.dataset == 'celeba':
        acc = per_gc_hits.sum() / per_gc_num.sum()

    elif args.dataset == 'split_cifar_100s':
        assert skew_ratio is not None
        per_gc_weight = np.zeros((2, 10))
        per_gc_weight[0, :5] = skew_ratio
        per_gc_weight[0, 5:] = 1 - skew_ratio
        per_gc_weight[1, :5] = 1 - skew_ratio
        per_gc_weight[1, 5:] = skew_ratio
        acc = (per_gc_hits / per_gc_num) * per_gc_weight * 0.1
        acc = acc.sum()

    elif args.dataset == 'split_imagenet_100c':
        noise_ratio = dataset.noise_ratio
        num_classes = dataset.num_classes
        biased_class = dataset.biased_class
        bsr = dataset.biased_skew_ratio
        acc = per_gc_hits / per_gc_num
        acc[0, :] = acc[0, :] * (1 - noise_ratio)
        acc[1, :] = acc[1, :] * noise_ratio
        acc[0, biased_class] = per_gc_hits[0, biased_class] / per_gc_num[0, biased_class] * (1 - bsr)
        acc[1, biased_class] = per_gc_hits[1, biased_class] / per_gc_num[1, biased_class] * bsr
        acc = (acc / num_classes).sum()
    else:
        raise NotImplementedError()

    return acc


def make_log_name(args):
    log_name = 'pretrained_' if args.pretrained else ''
    log_name += 'seed{}'.format(args.seed)
    log_name += '_lr{}_batch{}_epoch{}_decay{}'.format(args.lr, args.bs, args.epochs, args.decay)

    reg_based_methods = ['ewc', 'mas', 'lwf']
    rehearsal_based_methods = ['er', 'er_lwf', 'icarl', 'eeil', 'ssil', 'gdumb', 'der', 'bgs',
                               'er_bgs', 'er_lwf_bgs', 'icarl_bgs', 'eeil_bgs', 'ssil_bgs', 'der_bgs',
                               'groupdro_eeil', 'packnet_bgs']

    if args.trainer in reg_based_methods:
        log_name += '_lamb{}'.format(args.lamb)

    elif args.trainer in rehearsal_based_methods:
        if args.buffer_size is not None:
            log_name += '_buffer_size{}'.format(args.buffer_size)
        else:
            log_name += '_buffer_ratio{}'.format(args.buffer_ratio)

    elif args.trainer == 'packnet':
        log_name += '_prune_ratio{}'.format(args.prune_ratio)

    if 'groupdro' in args.trainer:
        log_name += '_gamma{}'.format(args.gamma)
        if args.trainer == 'groupdro_lwf':
            log_name += '_lamb{}'.format(args.lamb)

    return log_name


def list_dir(root: str, prefix: bool = False) -> List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def load_module(file_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_bmr(trainer, model, dataset, t, args, cur_t=None):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    dataloader = torch.utils.data.DataLoader(dataset, 2, drop_last=False,
                                             shuffle=False, **kwargs)
    num_classes = dataset.num_classes
    model.eval()
    if 'packnet' in args.trainer:
        weight_dict = trainer.pruner.apply_mask(model, t+1)

    with torch.no_grad():
        num_as = 0.
        num_hits = 0.
        for data in dataloader:
            input, group, label, _, _ = data
            input = input.cuda()
            label = label.cuda()

            if args.continual == 'task':
                output = model(input)[t]
            else:
                assert cur_t is not None
                output = []
                model_output = model(input)
                learned_tasks = cur_t + 1
                for _t in range(learned_tasks):
                    output.append(model_output[_t])
                output = torch.cat(output, dim=1)

                label = label + int(num_classes * t)

            preds = torch.argmax(output, 1)

            num_hits += (preds == label).sum()
            num_as += 1 if (preds == label).sum() == 1 else 0

        bmr = 0. if num_hits == 0. else num_as / num_hits

        if 'packnet' in args.trainer:
            trainer.pruner.restore_weight(model, weight_dict)

    return bmr



