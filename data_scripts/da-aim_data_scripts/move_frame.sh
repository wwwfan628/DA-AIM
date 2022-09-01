#!/bin/bash

#SBATCH --mail-type=ALL            # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=./%j.out          # where to store the output ( %j is the JOBID )
#SBATCH --error=./%j.err           # where to store error messages

scp -r /srv/beegfs02/scratch/da_action/data/kinetics/images/* /srv/beegfs02/scratch/da_action/data/kinetics-cad
now=$(date +"%T")
echo "finish time: $now"