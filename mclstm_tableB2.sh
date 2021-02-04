#/usr/bin/env sh

EXPNAME="function_static"  # hard coded in R-scripts...

### MC-FC ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_static.py \
  --operation add \
  --layer-type linear \
  --first-layer MCFC \
  --learning-rate 1e-4 \
  --max-iterations 500000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_static.py \
  --operation sub \
  --layer-type linear \
  --first-layer MCFC \
  --learning-rate 1e-4 \
  --hidden-size 3 \
  --regualizer 0 \
  --max-iterations 500000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_static.py \
  --operation mul \
  --layer-type MulMCFC \
  --first-layer MCFC \
  --hidden-size 3 \
  --learning-rate 1e-2 \
  --regualizer 0 \
  --max-iterations 3000000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data

### NAU / NMU (from Madsen et al.) ###
## NAC (from Madsen et al.) ###
### NALU (from Madsen et al.) ###

### create Madsen table ###
python export/simple_function_static.py \
  --tensorboard-dir "tensorboard/$EXPNAME/" \
  --csv-out "$EXPNAME.csv"
Rscript -e "source('export/function_task_static_mse_expectation.r')"
Rscript -e "source('export/function_task_static.r')"
