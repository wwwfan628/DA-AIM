
bsub -n 96 -R "rusage[mem=5000,ngpus_excl_p=8,scratch=5000]" -W 360:00 < experiments/supervised/bsub_mit.sh

bsub -n 128 -R "rusage[mem=3700,ngpus_excl_p=8,scratch=3000]" -R "select[gpu_model0==TITANRTX]" -W 1:00 < bsub_mit.sh

bsub -n 48 -R "rusage[mem=7500,ngpus_excl_p=8,scratch=6000]" -W 48:00 < bsub_R50_ava.sh

bsub -n 48 -R "rusage[mem=7500,ngpus_excl_p=8,scratch=15000]" -W 48:00 < bsub_R50_kin.sh

bsub -n 48 -R "rusage[mem=7500,ngpus_excl_p=8,scratch=15000]" -W 60:00 < bsub_R50_avakin.sh

bsub -n 48 -R "rusage[mem=7500,ngpus_excl_p=8,scratch=15000]" -W 72:00                                                                               
bsub -n 48 -R "rusage[mem=7500,ngpus_excl_p=8,scratch=15000]" -W 36:00 < bsub_R50_kin.sh           
bsub -n 48 -R "rusage[mem=7500,ngpus_excl_p=8,scratch=15000]" -W 48:00 < experiments/da/bsub_R50_kin_2_ava.sh