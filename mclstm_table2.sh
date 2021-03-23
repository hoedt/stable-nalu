#/usr/bin/env sh

# hardware
# CPU - 2x 18 cores @2.7Ghz or 2x 12 cores @3Ghz
# RAM - 384GB
# GPU - 1080Ti or Titan V

EXPNAME="function_recurrent"  # hard-coded in R-scripts...

### MC-LSTM ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation add \
  --layer-type MCLSTM \
  --learning-rate 1e-2 \
  --l2-out 1e-3 \
  --max-iterations 2000000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation sub \
  --layer-type MCLSTM \
  --hidden-size 3 \
  --learning-rate 1e-2 \
  --regualizer 0 \
  --l2-out 1e-4 \
  --max-iterations 500000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation mul \
  --layer-type MulMCLSTM \
  --hidden-size 3 \
  --learning-rate 5e-2 \
  --regualizer 0 \
  --max-iterations 2000000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data

### LSTM ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation add \
  --layer-type LSTM \
  --learning-rate 1e-2 \
  --l2-out 1e-4 \
  --max-iterations 2000000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation sub \
  --layer-type LSTM \
  --hidden-size 10 \
  --learning-rate 1e-2 \
  --regualizer 0 \
  --l2-out 1e-4 \
  --max-iterations 2000000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation mul \
  --layer-type LSTM \
  --hidden-size 10 \
  --learning-rate 1e-2 \
  --regualizer 0 \
  --max-iterations 2000000 \
  --name-prefix "$EXPNAME" \
  --seed {} \
  --remove-existing-data

### NAU / NMU ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation add \
  --layer-type ReRegualizedLinearNAC \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation sub \
  --layer-type ReRegualizedLinearNAC \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation mul \
  --layer-type ReRegualizedLinearNAC \
  --nac-mul mnac \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data

## NAC ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation add \
  --layer-type NAC \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation sub \
  --layer-type NAC \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation mul \
  --layer-type NAC \
  --nac-mul normal \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data

### NALU ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation add \
  --layer-type NALU \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation sub \
  --layer-type NALU \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/simple_function_recurrent.py \
  --operation mul \
  --layer-type NALU \
  --max-iterations 2000000 \
  --seed {} \
  --name-prefix "$EXPNAME" \
  --remove-existing-data

### create Madsen table ###
python export/simple_function_recurrent.py \
  --tensorboard-dir "tensorboard/$EXPNAME/" \
  --csv-out "results/$EXPNAME.csv"
Rscript -e "source('export/function_task_recurrent_mse_expectation.r')"
Rscript -e "source('export/function_task_recurrent.r')"
