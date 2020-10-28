# D-score
<img src='D-score-architecture.jpg'></img>

# Prerequisite 

## Resources

### D-score checkpoints
https://drive.google.com/drive/folders/18HHbd3kt3b1xc7_QCabKeWW2Dfyj5GSf?usp=sharing

### Finetuned LM
https://drive.google.com/drive/folders/1WMF3FOXexb_D0U2WflRu066O622kxaq3?usp=sharing

## Traing Procedure of D-score
D-score is trained with high-quality human-human conversations. Even though for our experiments, we used
the DSTC6 Customer Suppport, DSTC7 Knowledge-grounding and PERSONA-CHAT datasets, the framework is can be
also applied in other domains.

### Prerequisties
```
tensorflow=v1.15
best_checkpoint_copier
transformers=v2.11.0
```
### Usage	
`python main.py`
```
arguments:
  --data_dir {Path To Data Directory} \
  --roberta_config_file {Path To Roberta Base Model Config File} \
  --output_dir {Path To Save Checkpoints and Intermediate Files} \
  --corpus_name {Name of Corpus for Training: persona | dstc6 | dstc7} \
  --init checkpoint {Path To Finetuned LM} \
  --dropout_rate {default: 0.5} \
  --l2_reg_lambda {default: 0.1} \
  --batch_size {default: 8} \
  --dupe_factor {default: 5} \
  --max_pre_len {max length of previous context} \
  --max_post_len {max length of succeeding context} \
  --max_seq_len {max length of the current response} \
  --window_size {the value of K, actually here K refers to the total number of utterances including the pre-, post- and current utterances} \
  --lstm_size {default: 300} \
  --keep_checkpoint_max {default: 5} \
  --do_train \
  --do_eval \
  --learning_rate {default:1e-5}

## Instructions to Conduct Evaluation with D-score

