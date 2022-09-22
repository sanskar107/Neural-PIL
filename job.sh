#!/bin/bash
#SBATCH --job-name=neuralpil
#SBATCH --qos=normal
#SBATCH -p g24
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output="./logs/backpack_373_41833_83497-%j.log"
#SBATCH --open-mode=append


python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/backpack_373_41833_83497 --basedir holdout_16 --expname backpack_373_41833_83497 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/handbag_396_49743_97969 --basedir holdout_16 --expname handbag_396_49743_97969 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/handbag_399_51220_100573 --basedir holdout_16 --expname handbag_399_51220_100573 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/hydrant_106_12677_24990 --basedir holdout_16 --expname hydrant_106_12677_24990 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/hydrant_268_28513_58093 --basedir holdout_16 --expname hydrant_268_28513_58093 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/motorcycle_362_38283_75358 --basedir exp_debug_motorcycle_1gpu_allholdout --expname motorcycle_362_38283_75358 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/plant_253_27235_55344 --basedir holdout_16 --expname plant_253_27235_55344 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/plant_372_40884_81286 --basedir holdout_16 --expname plant_372_40884_81286 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/toytruck_353_37431_70460 --basedir holdout_16 --expname toytruck_353_37431_70460 --config configs/neural_pil/real_world.txt --rwfactor 2
# python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/toytruck_379_44672_89080 --basedir holdout_16 --expname toytruck_379_44672_89080 --config configs/neural_pil/real_world.txt --rwfactor 2
#python train_neural_pil.py --datadir /export/work/sanskara/svbrdf/data/NeRD/co3d/NeRD/NeRD/vase_380_44868_89574 --basedir holdout_16 --expname vase_380_44868_89574 --config configs/neural_pil/real_world.txt --rwfactor 2


# python train_neural_pil.py --datadir [DIR_TO_DATASET_FOLDER] --basedir [TRAIN_DIR] --expname [EXPERIMENT_NAME] --gpu [COMMA_SEPARATED_GPU_LIST]
