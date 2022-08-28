#!/bin/bash

#SBATCH --mail-type=ALL            # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=./%j.out          # where to store the output ( %j is the JOBID )
#SBATCH --error=./%j.err           # where to store error messages

scp -r /srv/beegfs02/scratch/da_action/data/new_datasets_may_2022/AAD/trimmed_images_may_18_2022_fps25/* \
 /srv/beegfs02/scratch/da_action/data/kinetics/images
now=$(date +"%T")
echo "finish time: $now"