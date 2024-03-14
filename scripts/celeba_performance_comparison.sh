# A sample script for performance comparison of CL methods on CelebA. Note that our results in the paper are the averaged results over 4 different seeds.
DEFAULT="--bs 128 --epochs 50 --save-model --n-tasks 8 --start-task 0 --dataset celeba --lr 0.001 --decay 0.01 --continual domain --optim adamw --seed 0"

# for fine-tuning, lwf, ewc, you can train bgs as below
# please change the trainer for other CL baselines {vanilla (for fine-tuning), lwf, ewc},
# we chose the hyperparmeters for CL methods based on the average accuracy up to third task
python main.py --trainer lwf --lamb 3 $DEFAULT
# For bgs, the trained modelpath after learning all tasks by a CL method is needed
python main.py --trainer bgs --buffer-size 160 --modelpath ./trained_models/none/celeba/domain/lwf/seed0_lr0.001_batch128_epoch50_decay0.01_lamb3.0_task7_st0_nt8.pt $DEFAULT

python main.py --trainer er --buffer-size 160 $DEFAULT
python main.py --trainer er_bgs --buffer-size 160 $DEFAULT

python main.py --trainer groupdro_lwf --gamma 0.03 --lamb 3.0 $DEFAULT