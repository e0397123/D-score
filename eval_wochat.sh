#!/bin/bash

srun -n 1 -w hlt02 --gres=gpu:1 python main.py \
	--data_dir ../datasets/wochat/ \
	--roberta_config_file ../pretrain_module/roberta/roberta-base-pretrained-tf/roberta-base-config.json \
	--corpus_name wochat \
	--output_dir test_dmtpe/ \
	--init_checkpoint /home/chen/projects/AFC_framework/pretrain_module/bert/run_pretrain/model.ckpt-100000 \
	--dropout_rate 0.3 \
	--l2_reg_lambda 0.05 \
	--batch_size=6 \
	--max_pre_len=50 \
	--max_post_len=50 \
	--window_size 3 \
	--lstm_size 300 \
	--do_predict
