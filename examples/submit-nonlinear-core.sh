#!/bin/sh

python3 classify.py  \
--dataset="synthetic_nonlinear" \
--architecture='nonlinear' \
--batch_size=120 \
--no-bn \
--no-cfl_annealing \
--cfl_rate_rise_factor=0.01 \
--cfl_rate_rise_time=30000 \
--classifier='counterfactual' \
--decay_rate=0.9999 \
--eval_batch_size=200 \
--fid=1 \
--data_path='data/' \
--h1=500 \
--h2=500 \
--no-img_data \
--lambda_reg=0.001 \
--learning_rate_class=0.001 \
--log_dir='test/' \
--n_input=2 \
--do-not-normalize \
--num_classes=2 \
--num_epochs_class=30 \
--no-regression \
--save_every=1000 \
--weight_countfact_loss=1
