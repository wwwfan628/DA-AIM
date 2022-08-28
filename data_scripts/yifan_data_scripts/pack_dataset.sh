#!/bin/bash

#SBATCH --mail-type=ALL            # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=./%j.out          # where to store the output ( %j is the JOBID )
#SBATCH --error=./%j.err           # where to store error messages

#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/ava_6_5000_all/annotations/ava_train_v2.2.csv', \
#'/srv/beegfs02/scratch/da_action/data/datasets_yifan/ava_6_5000_all/annotations/ava_val_v2.2.csv']" \
#--data_dir=/srv/beegfs02/scratch/da_action/data/ava/frames --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_ava_6_5000_all \
#--num_batchs=60     # ava_6_5000_all

#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_6_5000_all/annotations/kinetics_train.csv', \
#'/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_6_5000_all/annotations/kinetics_val.csv']" \
#--data_dir=/srv/beegfs02/scratch/da_action/data/kinetics/images --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_kinetics_6_5000_all \
#--num_batchs=64    # kin_6_5000_all
#
#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_3_5000_all/annotations/kinetics_train.csv', \
#'/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_3_5000_all/annotations/kinetics_val.csv']" \
#--data_dir=/srv/beegfs02/scratch/da_action/data/kinetics/images --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_kinetics_3_5000_all \
#--num_batchs=64    # kin_3_5000_all
#
#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/armasuisse_3_5000_all/annotations/armasuisse_train_v2.1.csv', \
#'/srv/beegfs02/scratch/da_action/data/datasets_yifan/armasuisse_3_5000_all/annotations/armasuisse_val_v2.1.csv']" \
#--data_dir=/srv/beegfs02/scratch/da_action/data/armasuisse/img --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_armasuisse_3_5000_all \
#--num_batchs=1    # armasuisse

#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/cad1/annotations/cad1_train_new.csv', \
#                                      '/srv/beegfs02/scratch/da_action/data/datasets_yifan/cad1/annotations/cad1_val_new.csv']" \
#                       --data_dir=/srv/beegfs02/scratch/da_action/data/new_datasets_may_2022/AAD/trimmed_images_may_18_2022_fps25 \
#                       --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_cad1_8_all_all \
#                       --num_batchs=1   # cad1
#
#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/cad2/annotations/cad2_train.csv', \
#                                      '/srv/beegfs02/scratch/da_action/data/datasets_yifan/cad2/annotations/cad2_val.csv']" \
#                       --data_dir=/srv/beegfs02/scratch/da_action/data/new_datasets_may_2022/CAD/camera-2-images-trimmed-fps-30 \
#                       --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_cad2_8_all_all \
#                       --num_batchs=1   # cad2
#
#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_8_5000_all/annotations/kinetics_train.csv', \
#                                      '/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_8_5000_all/annotations/kinetics_val.csv']" \
#                       --data_dir=/srv/beegfs02/scratch/da_action/data/kinetics/images \
#                       --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_kinetics_8_5000_all \
#                       --num_batchs=64   # kinetics_8_5000_all
#
#python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_8_5000_all+cad2/annotations/kinetics+cad2_train.csv', \
#                                      '/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_8_5000_all+cad2/annotations/kinetics+cad2_val.csv']" \
#                       --data_dir=/srv/beegfs02/scratch/da_action/data/kinetics/images \
#                       --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_kinetics_8_5000_all+cad2 \
#                       --num_batchs=64   # kinetics_8_5000_all+cad2

python pack_dataset.py --annotation="['/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_8_5000_all+cad1/annotations/kinetics+cad1_train.csv', \
                                      '/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_8_5000_all+cad1/annotations/kinetics+cad1_val.csv']" \
                       --data_dir=/srv/beegfs02/scratch/da_action/data/kinetics/images \
                       --tar_dir=/srv/beegfs02/scratch/da_action/data/datasets_yifan/tar_kinetics_8_5000_all+cad1 \
                       --num_batchs=64   # kinetics_8_5000_all+cad1