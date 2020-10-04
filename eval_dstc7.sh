#!/bin/bash

#SBATCH --job-name=eval_dstc7
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=eval_dstc7.log

/home/chen/anaconda3/envs/tensorflow-1.15/bin/python main.py \
	--data_dir datasets/dstc7  \
	--roberta_config_file /home/chen/projects/HADE/library/roberta/roberta-base-config.json \
	--output_dir run_dstc7/ \
	--corpus_name dstc7 \
	--init_checkpoint /home/chen/projects/HADE/finetuned/run_dstc7/model.ckpt-420000 \
	--dropout_rate 0.5 \
	--l2_reg_lambda 0.1 \
	--batch_size=8 \
	--max_pre_len=60 \
	--max_post_len=60 \
	--window_size 3 \
	--lstm_size 300 \
	--eval_type eval \
	--do_predict

