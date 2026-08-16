#!/bin/bash
#SBATCH --job-name=demand

#SBATCH --output=/home/users/weijia_6/Yangtze_code/logs/dengue_%A_%a.out
#SBATCH --error=/home/users/weijia_6/Yangtze_code/logs/dengue_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard        # Jasmin partition — change to lotus/high-mem if needed
#SBATCH --account=cpdn_nonnerc
#SBATCH --qos=long
 
 
module load jaspy  # or whichever R module is available on JASMIN
 
python test.py
