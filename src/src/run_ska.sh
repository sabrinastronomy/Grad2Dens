#!/bin/bash
#SBATCH --job-name=run_ska
#SBATCH --gres=gpu
#SBATCH --time=5:00:00
#SBATCH --mem=100G
#SBATCH -D /fred/oz113/sberger/paper_1_density/Grad2Dens/src
#SBATCH -o mesh_ska.out
#SBATCH -e mesh_ska.err
#SBATCH --mail-user=sabrinaberger55@gmail.com
#SBATCH --mail-type=ALL

source /home/sberger/.bash_profile

cd "/fred/oz113/sberger/paper_1_density/Grad2Dens"
conda deactivate
conda activate "/fred/oz113/sberger/"

python /fred/oz113/sberger/paper_1_density/Grad2Dens/src/alternating.py --ska_effect