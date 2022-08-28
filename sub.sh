# cd /srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan
# rsync -rav eu:/cluster/work/cvl/susaha/experiments/experiments_yifan/ experiments_yifan_euler/

rm -r /cluster/work/cvl/susaha/experiments/experiments_yifan/*    # delete all output directories


######################### Baseline #############################
#bsub -n 32 -J kin2AvaBaselinelr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_baseline/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaBaselinelr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_baseline/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2AsBaselinelr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_baseline/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2AsBaselinelr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_baseline/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#
#
#
######################### DACS #############################
#bsub -n 32 -J kin2Avalr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2Avalr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2Avalr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_lr_0.05.sh
#
#bsub -n 32 -J kin2Avalr0.01threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.01.sh
#
#bsub -n 32 -J kin2Avalr0.025threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.025.sh
#
#bsub -n 32 -J kin2Avalr0.05threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.05.sh
#
#bsub -n 32 -J kin2Avalr0.01mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.01.sh
#
#bsub -n 32 -J kin2Avalr0.025mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.025.sh
#
#bsub -n 32 -J kin2Avalr0.05mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.05.sh
#
#
#bsub -n 32 -J kin2Aslr0.01mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.01.sh
#
#bsub -n 32 -J kin2Aslr0.025mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.025.sh
#
#bsub -n 32 -J kin2Aslr0.05mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.05.sh
#
#bsub -n 32 -J kin2Aslr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2Aslr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#bsub -n 32 -J kin2Aslr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.05.sh
#
#bsub -n 32 -J kin2Aslr0.01threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.01.sh
#
#bsub -n 32 -J kin2Aslr0.025threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.025.sh
#
#bsub -n 32 -J kin2Aslr0.05threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.05.sh
#
#
#
#
######################### Class Mix #############################
#bsub -n 32 -J kin2AvaClassMixlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaClassMixlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2AvaClassMixlr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix/SLOWFAST_32x2_R50_KIN2AVA_lr_0.05.sh
#
#bsub -n 32 -J kin2AsClassMixlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2AsClassMixlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#bsub -n 32 -J kin2AsClassMixlr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.05.sh
#
#
#
#
######################### Pseudo Label #############################
#bsub -n 32 -J kin2AvaPesudoLabellr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_lr_0.05.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.01mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.025mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.025.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.05mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.05.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.01threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.025threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.025.sh
#
#bsub -n 32 -J kin2AvaPesudoLabellr0.05threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.05.sh
#
#
#bsub -n 32 -J kin2AsPesudoLabellr0.01mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.01.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.025mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.025.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.05mse -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.05.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.05.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.01threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.01.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.025threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.025.sh
#
#bsub -n 32 -J kin2AsPesudoLabellr0.05threshold -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_pseudo_label/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.05.sh
#
#
#
#
############################ DACS Resize ##################################
#bsub -n 32 -J kin2Avalr0.01Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2Avalr0.025Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2Avalr0.05Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_lr_0.05.sh
#
#bsub -n 32 -J kin2Avalr0.01thresholdResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.01.sh
#
#bsub -n 32 -J kin2Avalr0.025thresholdResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.025.sh
#
#bsub -n 32 -J kin2Avalr0.05thresholdResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_threshold_lr_0.05.sh
#
#bsub -n 32 -J kin2Avalr0.01mseResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.01.sh
#
#bsub -n 32 -J kin2Avalr0.025mseResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.025.sh
#
#bsub -n 32 -J kin2Avalr0.05mseResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2AVA_mse_lr_0.05.sh
#
#
#bsub -n 32 -J kin2Aslr0.01mseResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.01.sh
#
#bsub -n 32 -J kin2Aslr0.025mseResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.025.sh
#
#bsub -n 32 -J kin2Aslr0.05mseResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_mse_lr_0.05.sh
#
#bsub -n 32 -J kin2Aslr0.01Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2Aslr0.025Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#bsub -n 32 -J kin2Aslr0.05Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.05.sh
#
#bsub -n 32 -J kin2Aslr0.01thresholdResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.01.sh
#
#bsub -n 32 -J kin2Aslr0.025thresholdResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.025.sh
#
#bsub -n 32 -J kin2Aslr0.05thresholdResize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_dacs_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_threshold_lr_0.05.sh
#
#
#
#
############################ Class Mix Resize ##################################
#bsub -n 32 -J kin2AvaClassMixlr0.01Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix_resize/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaClassMixlr0.025Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix_resize/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2AvaClassMixlr0.05Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix_resize/SLOWFAST_32x2_R50_KIN2AVA_lr_0.05.sh
#
#bsub -n 32 -J kin2AsClassMixlr0.01Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2AsClassMixlr0.025Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#bsub -n 32 -J kin2AsClassMixlr0.05Resize -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_class_mix_resize/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.05.sh
#
#
#
#
############################# Clip Order ##################################
#bsub -n 32 -J kin2AvaClipOrderlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_clip_order/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaClipOrderlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_clip_order/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2AsClipOrderlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_clip_order/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2AsClipOrderlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_clip_order/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#
#
#
############################# Rotation ##################################
#bsub -n 32 -J kin2AvaRotationlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_rotation/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaRotationlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_rotation/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2AsRotationlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_rotation/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2AsRotationlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_rotation/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#
#
#
############################# GRL ##################################
#bsub -n 32 -J kin2AvaGRLlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_grl/SLOWFAST_32x2_R50_KIN2AVA_lr_0.01.sh
#
#bsub -n 32 -J kin2AvaGRLlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_grl/SLOWFAST_32x2_R50_KIN2AVA_lr_0.025.sh
#
#bsub -n 32 -J kin2AsGRLlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_grl/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J kin2AsGRLlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/da_grl/SLOWFAST_32x2_R50_KIN2ARMASUISSE_lr_0.025.sh
#



############################################## NEW DATASETS ##############################################

############################# Oracle ##################################
#bsub -n 32 -J CAD1lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/oracle/SLOWFAST_32x2_R50_CAD1_lr_0.01.sh
#
#bsub -n 32 -J CAD1lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/oracle/SLOWFAST_32x2_R50_CAD1_lr_0.025.sh
#
#bsub -n 32 -J CAD1lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/oracle/SLOWFAST_32x2_R50_CAD1_lr_0.05.sh
#
#bsub -n 32 -J CAD2lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/oracle/SLOWFAST_32x2_R50_CAD2_lr_0.01.sh
#
#bsub -n 32 -J CAD2lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/oracle/SLOWFAST_32x2_R50_CAD2_lr_0.025.sh
#
#bsub -n 32 -J CAD2lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/oracle/SLOWFAST_32x2_R50_CAD2_lr_0.05.sh

############################# Baseline ##################################
#bsub -n 32 -J CAD22CAD1lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_CAD22CAD1_lr_0.01.sh
#
#bsub -n 32 -J CAD22CAD1lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_CAD22CAD1_lr_0.025.sh
#
#bsub -n 32 -J CAD22CAD1lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_CAD22CAD1_lr_0.05.sh
#
#bsub -n 32 -J KIN2CAD1lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KIN2CAD1_lr_0.01.sh
#
#bsub -n 32 -J KIN2CAD1lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KIN2CAD1_lr_0.025.sh
#
#bsub -n 32 -J KIN2CAD1lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KIN2CAD1_lr_0.05.sh
#
#bsub -n 32 -J KINCAD22CAD1lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KINCAD22CAD1_lr_0.01.sh
#
#bsub -n 32 -J KINCAD22CAD1lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KINCAD22CAD1_lr_0.025.sh
#
#bsub -n 32 -J KINCAD22CAD1lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KINCAD22CAD1_lr_0.05.sh

#bsub -n 32 -J CAD12CAD2lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_CAD12CAD2_lr_0.01.sh

#bsub -n 32 -J CAD12CAD2lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_CAD12CAD2_lr_0.025.sh

bsub -n 32 -J CAD12CAD2lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_CAD12CAD2_lr_0.05.sh

#bsub -n 32 -J KIN2CAD2lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KIN2CAD2_lr_0.01.sh

#bsub -n 32 -J KIN2CAD2lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KIN2CAD2_lr_0.025.sh

bsub -n 32 -J KIN2CAD2lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KIN2CAD2_lr_0.05.sh

#bsub -n 32 -J KINCAD12CAD2lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KINCAD12CAD2_lr_0.01.sh
#
#bsub -n 32 -J KINCAD12CAD2lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KINCAD12CAD2_lr_0.025.sh
#
#bsub -n 32 -J KINCAD12CAD2lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_baseline/SLOWFAST_32x2_R50_KINCAD12CAD2_lr_0.05.sh

############################# DACS ##################################
#bsub -n 32 -J CAD22CAD1_DACS_mse_lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_CAD22CAD1_mse_lr_0.01.sh
#
#bsub -n 32 -J CAD22CAD1_DACS_mse_lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_CAD22CAD1_mse_lr_0.025.sh
#
#bsub -n 32 -J CAD22CAD1_DACS_mse_lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_CAD22CAD1_mse_lr_0.05.sh
#
#bsub -n 32 -J KIN2CAD1_DACS_mse_lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KIN2CAD1_mse_lr_0.01.sh
#
#bsub -n 32 -J KIN2CAD1_DACS_mse_lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KIN2CAD1_mse_lr_0.025.sh
#
#bsub -n 32 -J KIN2CAD1_DACS_mse_lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KIN2CAD1_mse_lr_0.05.sh
#
#bsub -n 32 -J KINCAD22CAD1_DACS_mse_lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KINCAD22CAD1_mse_lr_0.01.sh
#
#bsub -n 32 -J KINCAD22CAD1_DACS_mse_lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KINCAD22CAD1_mse_lr_0.025.sh
#
#bsub -n 32 -J KINCAD22CAD1_DACS_mse_lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KINCAD22CAD1_mse_lr_0.05.sh

bsub -n 32 -J CAD12CAD2_DACS_mse_lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_CAD12CAD2_mse_lr_0.01.sh

#bsub -n 32 -J CAD12CAD2_DACS_mse_lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_CAD12CAD2_mse_lr_0.025.sh

#bsub -n 32 -J CAD12CAD2_DACS_mse_lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_CAD12CAD2_mse_lr_0.05.sh

bsub -n 32 -J KIN2CAD2_DACS_mse_lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KIN2CAD2_mse_lr_0.01.sh

#bsub -n 32 -J KIN2CAD2_DACS_mse_lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KIN2CAD2_mse_lr_0.025.sh

#bsub -n 32 -J KIN2CAD2_DACS_mse_lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KIN2CAD2_mse_lr_0.05.sh

#bsub -n 32 -J KINCAD12CAD2_DACS_mse_lr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KINCAD12CAD2_mse_lr_0.01.sh
#
#bsub -n 32 -J KINCAD12CAD2_DACS_mse_lr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KINCAD12CAD2_mse_lr_0.025.sh
#
#bsub -n 32 -J KINCAD12CAD2_DACS_mse_lr0.05 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/new_datasets/da_dacs/SLOWFAST_32x2_R50_KINCAD12CAD2_mse_lr_0.05.sh


############################# Oracle ##################################
#bsub -n 32 -J AVAlr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/oracle/SLOWFAST_32x2_R50_AVA_lr_0.01.sh
#
#bsub -n 32 -J AVAlr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/oracle/SLOWFAST_32x2_R50_AVA_lr_0.025.sh
#
#bsub -n 32 -J ARMASUISSElr0.01 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/oracle/SLOWFAST_32x2_R50_ARMASUISSE_lr_0.01.sh
#
#bsub -n 32 -J ARMASUISSElr0.025 -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/oracle/SLOWFAST_32x2_R50_ARMASUISSE_lr_0.025.sh



############################# CONFUSION MATRIX ##################################
#bsub -n 32 -J DACS_CM -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/kincad22cad1_plot_cm/SLOWFAST_32x2_R50_KINCAD22CAD1_DACS_CM.sh
#
#bsub -n 32 -J PLABEL_CM -R "rusage[mem=8000,ngpus_excl_p=4,scratch=15000]" -R "span[hosts=1]" -R "select[gpu_mtotal0>=20040]" -W 24:00 < experiments/gurkirt_and_suman/kincad22cad1_plot_cm/SLOWFAST_32x2_R50_KINCAD22CAD1_PSEUDOLABEL_CM.sh