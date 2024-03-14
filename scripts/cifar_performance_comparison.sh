# A sample script for performance comparison of CL methods on Split CIFAR-100S. Note that our results in the paper are the averaged results over 4 different seeds.
DEFAULT="--bs 256 --epochs 70 --save-model --n-tasks 10 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw --seed 0"

# For fine-tuning, lwf, ewc, you can train bgs as below
# Please change the trainer for other CL baselines {vanilla (for fine-tuning), lwf, ewc},
# We chose the hyperparmeters for CL methods based on the average accuracy up to third task
python main.py --trainer lwf --lamb 1 $DEFAULT
# For bgs, the trained modelpath after learning all tasks by a CL method is needed
python main.py --trainer bgs --buffer-size 1000 --modelpath ./trained_models/none/split_cifar_100s_skew_random/task/lwf/seed0_lr0.001_batch256_epoch70_decay0.01_lamb1.0_task9_st0_nt10.pt $DEFAULT

python main.py --trainer er --buffer-size 1000 $DEFAULT
python main.py --trainer er_bgs --buffer-size 1000 $DEFAULT

python main.py --trainer packnet --prune-ratio 0.3 $DEFAULT
python main.py --trainer packnet_bgs --buffer-size 1000 --modelpath ./trained_models/none/split_cifar_100s_skew_random/task/packnet/seed0_lr0.001_batch256_epoch70_decay0.01_prune_ratio\[0.3\]_task9_st0_nt10.pt $DEFAULT

python main.py --trainer groupdro_lwf --gamma 1e-4 --lamb 1.0 $DEFAULT