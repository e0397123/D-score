#!/bin/bash

#SBATCH --job-name=train_dd
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=train_dd.log

/home/chen/anaconda3/envs/tensorflow-1.15/bin/python main.py \
	--data_dir datasets/daily-dialog  \
	--roberta_config_file /home/chen/projects/HADE/library/roberta/roberta-base-config.json \
	--output_dir run_dailydialog/ \
	--corpus_name DAILYD \
	--init_checkpoint /home/chen/projects/HADE/finetuned/run_dd/model.ckpt-100000 \
	--dropout_rate 0.5 \
	--l2_reg_lambda 0.1 \
	--batch_size=8 \
	--dupe_factor=5 \
	--max_pre_len=60 \
	--max_post_len=60 \
	--window_size 3 \
	--lstm_size 300\
	--keep_checkpoint_max 5 \
	--do_train \
	--do_eval
