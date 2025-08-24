
# change this variable to match your machine.
N_GPU=8


# Inpainting
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt inpaint-center
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt inpaint-freeform1020
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt inpaint-freeform2030



python train.py --n-gpu-per-node 1 --beta-max 1.0 --corrupt color
python train.py --n-gpu-per-node 2 --beta-max 1.0 --corrupt color --ckpt color

python train.py --n-gpu-per-node 2 --beta-max 1.0 --corrupt normal 
python train.py --n-gpu-per-node 4 --beta-max 1.0 --corrupt normal 
python train.py --n-gpu-per-node 8 --beta-max 1.0 --corrupt normal 
