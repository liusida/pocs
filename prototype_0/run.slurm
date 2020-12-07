#!/bin/bash
#SBATCH --job-name="SWARMS"
#SBATCH --time 0-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH --output=/users/d/m/dmatthe1/job_logs/%x.%A_%a.txt
#SBATCH --array=1-30

echo `hostname`

source ~/.bashrc

conda activate pocs
echo `printenv`
echo `which python`
echo `pwd`

cd $HOME/pocs/prototype_0

python main.py --raw --seed $SLURM_ARRAY_TASK_ID --steps $STEPS -p $POLICY -n $NUM_AGENTS

echo $SLURM_ARRAY_TASK_ID
echo "DONE"
