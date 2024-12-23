#!/usr/bin/env bash
#SBATCH --job-name=conda-test                 # nazwa zadania
#SBATCH --partition=gpu_spot                   # partycja (CPU/GPU)
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00                   # limit czasu wykonania zadania - 5min
#SBATCH --output=.results/conda-test-%A-%a.out            # schemat pliku wynikowego %j - numer zadania
#SBATCH --mail-type=ALL                   # konfiguracja powiadomienia e-mail
#SBATCH --mail-user=fp.patyk@gmail.com  # adres dla powiadomień e-mail
#SBATCH --mail-type=ARRAY_TASKS,END

# Przykład wykonania skryptu Pythona w trybie wsadowym Slurm

export WANDB_API_KEY=""


# aktywacja modułu anaconda
module load anaconda rclone
conda activate medicraft-dev

# przejście do katalogu roboczego
cd /work/$USER/python-test || exit -1


echo "Hello from the batch script"
# Wykonanie skryptu
srun python conda_test.py $SLURM_ARRAY_TASK_ID
