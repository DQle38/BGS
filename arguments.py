import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual Learning')
    # optimizing scheme
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='optimizer')
    parser.add_argument('--epochs', '--epoch', type=int, default=70, help='the number of epochs to train each task')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use imagenet pretrained model')
    parser.add_argument('--modelpath', default=None, type=str)
    parser.add_argument('--bs', type=int, default=256, metavar='N', help='batch size for training (default: 256)')
    parser.add_argument('--n-workers', type=int, default=4, help='number of thread')

    # Experiment arguments
    parser.add_argument('--seed', type=int, default=0,
                        help='seeds values to be used; seed introduces randomness by changing order of classes')
    parser.add_argument('--ex-id', type=str, default='none', help='experiment identifier for logger')
    parser.add_argument('--save-model',  default=False, action='store_true', help='save the learned model')    

    # continual learning related parameter
    parser.add_argument('--start-task', default=0, type=int, help='from which task to train the model')
    parser.add_argument('--n-tasks', default=None, type=int, help='the number of task to train')
    parser.add_argument('--task-seq', default=None, type=int, nargs='+', help='task-sequence')
    parser.add_argument('--continual', type=str, default='task', choices=['domain', 'class', 'task'],
                        help='continual learning scenario')

    parser.add_argument('--trainer', default='vanilla', type=str,
                        choices=['vanilla', 'freezing',
                                 'lwf', 'ewc', 'packnet', 'er', 'er_lwf', 'icarl', 'eeil', 'ssil',
                                 'er_bgs', 'er_lwf_bgs', 'icarl_bgs', 'eeil_bgs', 'ssil_bgs', 'packnet_bgs',
                                 'groupdro_lwf', 'groupdro_eeil', 'gdumb', 'bgs', 'der', 'der_bgs', 'mas'],
                        help='continual learning method')

    # dataset
    parser.add_argument('--dataset', default='split_cifar_100s', type=str,
                        choices=['split_cifar_100s', 'celeba_two', 'celeba', 'split_imagenet_100c'],
                        help='dataset to train')

    parser.add_argument('--skew-ratio', default=None, type=float, nargs='+',
                        help='dataset skewness')
    # Split CIFAR-100S
    parser.add_argument('--accumulation', default=False, action='store_true', help='accumulation experiments or not')
    parser.add_argument('--n-biased', default=0, type=int, help='number of biased tasks for accumulation experiments')
    parser.add_argument('--for-backward', default=False, action='store_true', help='accumulation backward')

    # Split ImageNet-100C
    parser.add_argument('--noise-type', default='gaussian', choices=['gaussian', 'frost'])

    # hyperparams
    parser.add_argument('--lamb', type=float, default=1.0, help='regularization strength')
    parser.add_argument('--temperature', type=float, default=2, help='temperature for distillation')

    # Group DRO
    parser.add_argument('--gamma', type=float, default=0.0, help='hyperparameter for Group Dro')

    # packnet
    parser.add_argument('--prune-ratio', type=float, default=0.5, nargs='+', help='pruning ratio for each layer')
    parser.add_argument('--post-prune-epochs', type=int, default=10, help='pruning epoch')

    # Rehearsal-based
    parser.add_argument('--buffer-ratio', default=0.1, type=float, required=False,
                        help='the proportion of training samples to set buffer size')
    parser.add_argument('--buffer-size', default=None, type=int, required=False, help='the memory capacity')

    # GDumb
    parser.add_argument('--use-cutmix', default=False, action='store_true',  help='use cutmix for gdumb')
    parser.add_argument('--cutmix-prob', default=0.5, type=float,  help='cutmix probability')
    parser.add_argument('--beta', type=float, default=1., help='beta for the distribution')

    args = parser.parse_args()
    return args
