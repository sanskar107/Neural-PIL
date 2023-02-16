#!/bin/bash
#SBATCH --job-name=porcelain_mug
#SBATCH -p g24
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output="./exp_logs/porcelain_mug-new-%a-%j.log"
#SBATCH --open-mode=append


scene="porcelain_mug"
expname="porcelain_mug"

rwfactor='2'

echo "====== Scene: $scene : $rwfactor ======"

datadir="/export/work/sanskar/data/exp_llff_data/$scene"
envdir="/export/work/sanskar/data/exp_llff_data/$scene/val_envmaps.npy"

# python train_neural_pil.py --datadir "$datadir" --basedir "new_out_objrel" --expname "$expname" --config configs/neural_pil/real_world.txt --rwfactor "$rwfactor" --single_env

# python train_neural_pil.py --datadir "$datadir" --basedir "out_objrel" --expname "$expname" --config configs/neural_pil/real_world.txt --rwfactor "$rwfactor" --single_env --render_only

python train_neural_pil.py --datadir "$datadir" --basedir "new_exp_out_objrel" --expname "$expname" --config configs/neural_pil/real_world.txt --rwfactor "$rwfactor" --single_env --envmap_path "$envdir"

# python train_neural_pil.py --datadir "$datadir" --basedir "exp_out_objrel" --expname "$expname" --config configs/neural_pil/real_world.txt --rwfactor "$rwfactor" --single_env --render_only --envmap_path "$envdir"
