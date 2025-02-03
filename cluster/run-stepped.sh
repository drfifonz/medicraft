#!/usr/bin/env bash
#SBATCH --job-name=gen-medicraft-stepped
#SBATCH --partition=gpu_spot
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=.run-logs/stepped/run-%A-%a.out
#SBATCH --mail-user=fp.patyk@gmail.com
#SBATCH --mail-type=ALL


source /home/fpatyk/.cache/pypoetry/virtualenvs/medicraft-7W8s1oxG-py3.11/bin/activate



cd /work/$USER/medicraft || exit -1


srun python src/medicraft/main.py  -f configs/train-generator-classed.yml --step $SLURM_ARRAY_TASK_ID
# srun python src/medicraft/main.py  -f configs/train-generator.yml --step $SLURM_ARRAY_TASK_ID




