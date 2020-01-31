import argparse
import os
import datetime
import mimesis


HEADER = "qsubmit -n bertner -o /u/qfeng/bertLogs -m 15G -t 24:00:00 -a `find_1080.sh` -gpu 1 /work/smt2/qfeng/Project/huggingface/hf_venv/bin/python3 \
/work/smt2/qfeng/Project/huggingface/transformers/qihui/bert_ner.py "
DEFAULT_SETTING = "\
--data_dir CoNLL \
--model_type bert \
--max_seq_length 128 \
--cache_dir base-uncased/ \
--gradient_accumulation_steps 4 \
--learning_rate 1e-4 \
--num_train_epochs 15 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 8 \
--logging_steps 1000 \
--save_steps 10000 \
--evaluate_during_training \
--test_during_training \
--do_train --do_eval --do_predict \
--seed 1 \
--yago_reference \
--weight_decay 0.01 \
--warmup_proportion 0.1 \
--do_significant_check \
--overwrite_cache \
--do_lower_case "

"""
--model_name_or_path bert-base-cased \
--additional_output_tag high_lr \
--overwrite_output_dir \
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--eval_ckp", type=str, default="")

    args = parser.parse_args()

    command = []

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    hashtag = mimesis.Cryptographic().token_hex(5)

    exp_folder = os.path.join("/work/smt3/wwang/TAC2019/qihui_data/homemade_bert/", args.exp_name)
    ckp_list = os.listdir(exp_folder)
    if args.eval_ckp == "":
        for ckp in ckp_list:
            args.model_name = os.path.join(exp_folder, ckp)
            output_tag = ckp
            settings = "--model_name_or_path {}".format(args.model_name)
            cmd = HEADER.replace('bertner', 'bertner_{}'.format(output_tag)) + DEFAULT_SETTING + settings
            command.append(cmd)
        with open(os.path.join('/work/smt2/qfeng/Project/huggingface/scripts', '{}_{}.sh'.format(date, hashtag)), 'w') as fout:
            fout.write('\n'.join(command))
            print(os.path.join('/work/smt2/qfeng/Project/huggingface/scripts', '{}_{}.sh'))
