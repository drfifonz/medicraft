#!/usr/bin/env bash
#SBATCH --job-name=medicraft-classification-multi  # nazwa zadania
#SBATCH --partition=gpu_spot                   # partycja (CPU/GPU)
#SBATCH --array=1-2                       # liczba etapów
#SBATCH --ntasks=4                        # 4 CPU
#SBATCH --time=00:30:00                   # limit czasu wykonania zadania - 30min
#SBATCH --output=.results/runs/run-%A-%a.out        # schemat pliku wynikowego %A - numer zadania, %a - numer etapu
#SBATCH --mail-type=ALL                   # konfiguracja powiadomienia e-mail
#SBATCH --mail-user=twoj_email@serwer.pl  # adres dla powiadomień e-mail

# Przykład wykonania skryptu Pythona w trybie wsadowym Slurm         

# aktywacja modułu anaconda
module load anaconda

conda activate medicraft 
# Wykonanie skryptu

export WANDB_API_KEY=<api-key>

# przejście do katalogu roboczego
cd /work/$USER/medicraft || exit -1



# Wykonanie skryptu
srun --mpi=pmi2 python src/medicraft/main.py -v -f classification.yml




