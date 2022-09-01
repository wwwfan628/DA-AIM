#!/bin/bash

#SBATCH --mail-type=ALL                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=./%j.out          # where to store the output ( %j is the JOBID )
#SBATCH --error=./%j.err           # where to store error messages

#python preprocess_dataset.py --dataset='ava' \
#--base_dir_source='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/ava/' \
#--base_dir_target='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/ava_0_0000_all/' \
#--num_samples=5000
python preprocess_dataset.py --dataset='kinetics' \
--base_dir_source='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/kinetics/' \
--base_dir_target='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/kinetics_0_0000_all/' \
--num_samples=5000
#python preprocess_dataset.py --dataset='armasuisse' \
#--base_dir_source='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/armasuisse/' \
#--base_dir_target='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/armasuisse_0_0000_all/' \
#--num_samples=5000