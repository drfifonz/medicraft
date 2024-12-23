#!/usr/bin/env bash
#SBATCH --job-name=gen-medicraft  # nazwa zadania
#SBATCH --partition=gpu_spot                   # partycja (CPU/GPU)
#SBATCH --array=1-5                     # liczba etapów
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00                   # limit czasu wykonania zadania - 30min
#SBATCH --output=.results/runs/run-%A-%a.out        # schemat pliku wynikowego %A - numer zadania, %a - numer etapu
#SBATCH --mail-user=fp.patyk@gmail.com  # adres dla powiadomień e-mail
#SBATCH --mail-type=ARRAY_TASKS,FAIL

# Przykład wykonania skryptu Pythona w trybie wsadowym Slurm         

# aktywacja modułu anaconda
export WANDB_API_KEY="s"
module load anaconda

conda activate medicraft-dev
# Wykonanie skryptu


# przejście do katalogu roboczego
cd /work/$USER/medicraft || exit -1


srun python src/generate_dataset_temp.py $SLURM_ARRAY_TASK_ID




