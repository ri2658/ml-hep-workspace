#!/bin/bash

models=("03_17_pyg_ef_nn_cw_8_2_noqcdlep_abseta" "03_17_pyg_ef_nn_cw_8_2_noqcdlep_100_pfcands" "03_17_pyg_ef_nn_cw_8_2_noqcdlep" "03_17_pyg_ef_nn_cw_1_1_noqcdlep")
configs=("ak15_4q3q_flat_eta_genHm_pt300_cw_8_2_noqcdlep_abseta" "ak15_4q3q_flat_eta_genHm_pt300_cw_8_2_noqcdlep_100_pfcands" "ak15_4q3q_flat_eta_genHm_pt300_cw_8_2_noqcdlep" "ak15_4q3q_flat_eta_genHm_pt300_cw_1_1_noqcdlep")
jobnames=("03-17-noqcdlep-abseta" "03-17-noqcdlep-100-pfcands" "03-17-noqcdlep" "03-17-noqcdlep-1-1")

for i in ${!models[@]}; do
  kubectl delete jobs weaver-job-rk-${jobnames[$i]}-inference -n cms-ml-hvv
  python3 inference_from_templ.py --model-name ${models[$i]} --data-config ${configs[$i]} --job-name ${jobnames[$i]} --overwrite --samples GluGluToBulkGravitonToHHTo4W_JHUGen_M-2500_narrow --sample-names bulkg_hsm
done
