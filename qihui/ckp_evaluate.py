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
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 15 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 8 \
--logging_steps 1000 \
--save_steps 10000 \
--evaluate_during_training \
--test_during_training \
--do_train --do_eval --do_predict \
--seed 1 \
--weight_decay 0.01 \
--warmup_proportion 0.1 \
--do_significant_check \
--overwrite_output_dir \
--overwrite_cache \
--yago_reference \
--do_lower_case "

"""

--model_name_or_path bert-base-uncased \
--additional_output_tag high_lr \
--overwrite_cache \
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
            if 'subtask' in ckp or 'test' in ckp:
                continue
            num_steps = int(ckp[len('checkpoint-step'):])
            if num_steps <= 520000:
                continue
            args.model_name = os.path.join(exp_folder, ckp)
            output_tag = ckp
            settings = "--model_name_or_path {} --additional_output_tag yago".format(args.model_name)
            cmd = HEADER.replace('bertner', 'bertner_{}'.format(output_tag)) + DEFAULT_SETTING + settings
            command.append(cmd)
        with open(os.path.join('/work/smt2/qfeng/Project/huggingface/scripts', '{}_{}.sh'.format(date, hashtag)), 'w') as fout:
            fout.write('\n'.join(command))
            print(os.path.join('/work/smt2/qfeng/Project/huggingface/scripts', '{}_{}.sh'.format(date, hashtag)))
