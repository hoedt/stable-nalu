#/usr/bin/env sh

EXPNAME="sequential_mnist_sum"

### MC-LSTM ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/sequential_mnist.py \
  --operation cumsum \
  --layer-type MCLSTM \
  --interpolation-length 10 \
  --extrapolation-lengths [1,10,100,200,300,400,500,600,700,800,900,1000] \
  --regualizer 0 \
  --mnist-regularizer 1e-4 \
  --seed {} \
  --max-epochs 1000 \
  --name-prefix sequential_mnist \
  --remove-existing-data

### NAU ###
seq 0 99 | xargs -n1 -P10 -i -- \
python experiments/sequential_mnist.py \
  --operation cumsum \
  --layer-type ReRegualizedLinearNAC \
  --interpolation-length 10 \
  --extrapolation-lengths [1,10,100,200,300,400,500,600,700,800,900,1000] \
  --regualizer-z 1 \
  --seed {} \
  --max-epochs 1000 \
  --name-prefix "$EXPNAME" \
  --remove-existing-data

### Reference ###
seq 0 9 | xargs -n1 -P10 -i -- \
python experiments/sequential_mnist.py \
  --operation cumsum \
  --layer-type ReRegualizedLinearNAC \
  --model-simplification solved-accumulator \
  --interpolation-length 10 \
  --extrapolation-lengths '[1,10,100,200,300,400,500,600,700,800,900,1000]' \
  --seed {} \
  --max-epochs 1000 \
  --name-prefix "${EXPNAME}_reference" \
  --remove-existing-data

### create figure ###
python export/sequential_mnist.py \
  --tensorboard-dir "tensorboard/${EXPNAME}_reference/" \
  --csv-out "results/${EXPNAME}_reference.csv"
python export/sequential_mnist.py \
  --tensorboard-dir "tensorboard/$EXPNAME/" \
  --csv-out "results/${EXPNAME}_long.csv"
Rscript -e "source('export/sequential_mnist_sum_long.r')"
