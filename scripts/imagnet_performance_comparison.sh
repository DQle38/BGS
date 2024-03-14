# A sample script for performance comparison of CL methods on CelebA. Note that our results in the paper are the averaged results over 4 different seeds.
DEFAULT="--bs 128 --epochs 70 --save-model --n-tasks 10 --start-task 0 --dataset split_imagenet_100c --lr 0.001 --decay 0.01 --noise-type gaussian --continual class --optim adamw --seed 0"

# for er, lwf + er, iCaRL, EEIL you can train bgs as below
# please change the trainer for other CL baselines {vanilla (for fine-tuning), lwf, ewc} (+ BGS),
# we chose the hyperparmeters for CL methods based on the average accuracy up to third task
python main.py --trainer er --buffer-size 1000 $DEFAULT
python main.py --trainer er_bgs --buffer-size 1000 $DEFAULT

python main.py --trainer groupdro_eeil --gamma 1e-4 --buffer-size 1000 $DEFAULT