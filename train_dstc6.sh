#!/bin/bash

#SBATCH --job-name=train_dstc6
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=train_dstc6.log

/home/chen/anaconda3/envs/tensorflow-1.15/bin/python main.py \
	--data_dir datasets/dstc6/ \
	--corpus_name dstc6 \
	--roberta_config_file /home/chen/projects/HADE/library/roberta/roberta-base-config.json \
	--output_dir run_dstc6/ \
	--init_checkpoint /home/chen/projects/HADE/finetuned/run_dstc6/model.ckpt-100000 \
	--dropout_rate 0.5 \
	--l2_reg_lambda 0.1 \
	--batch_size=8 \
	--max_pre_len=60 \
	--max_post_len=60 \
	--max_seq_len=30 \
	--window_size 3 \
	--lstm_size 300 \
	--do_train \
	--do_eval \
	--dupe_factor=5 \
	--keep_checkpoint_max 5
