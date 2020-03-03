from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import pickle
import sys

sys.path.append("/u/qfeng/Project/huggingface/transformers/")

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# import qihui
from qihui.model.utils_multi_bertner import convert_pickle_to_features, MultiNerDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer, BertForPreTraining
from qihui.model.modeling_multilabel_bertner import BertForMultipleLabelTokenClassification
from transformers.configuration_bert import BertMultipleLabelConfig
# from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
# from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer
# from transformers import CamembertConfig, CamembertForTokenClassification, CamembertTokenizer

logger = logging.getLogger(__name__)
REFERENCE_SIZE=7304
DEFAULT_DATA_REPO = '/work/wwang/qihui/yago/yago_se_dicts/'
DEFAULT_CACHE_REPO = '/work/smt2/qfeng/Project/huggingface/pretrain/'
DEFAULT_OUTPUT_REPO = '/work/smt3/wwang/TAC2019/qihui_data/multi_bertner/'

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, model, tokenizer, masked_token_label_id):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    assert(args.max_steps is not None and args.max_steps > 0)
    assert(args.num_train_epochs is not None and args.num_train_epochs > 0)

    t_total = args.max_steps
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)
    for epochs in train_iterator:
        subtask_iterator = trange(10, desc="Subtask", disable=args.local_rank not in [-1, 0])
        for subtask_id in subtask_iterator:
            pickle_file = os.path.join(DEFAULT_DATA_REPO, 'se_dict_{}.pickle'.format(str(subtask_id)))
            train_dataset = load_and_cache_examples(args, tokenizer, pickle_file, masked_token_label_id)
            # TODO:FEB27
            # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            batches = [[train_dataset[i][x:x + args.train_batch_size] for i in range(4)] for x in range(0, len(train_dataset[0]), args.train_batch_size)]
            random.shuffle(batches)

            # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
            epoch_iterator = tqdm(batches, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()
                # construct the ground truth type output
                # logger.info(len(batch))
                # logger.info(batch[0][0])
                # logger.info(batch[3][0])
                # assert (torch.sum(one_hot(torch.LongTensor(batch[3][0][2]),num_classes=REFERENCE_SIZE),dim=-2).size()== torch.zeros(REFERENCE_SIZE).size())
                # test = torch.stack([torch.sum(one_hot(torch.LongTensor(batch[3][0][p]),num_classes=REFERENCE_SIZE),dim=-2) if len(batch[3][0][p])>0 else torch.LongTensor([0]*REFERENCE_SIZE) for p in range(args.max_seq_length)])
                label_type_ids = torch.stack([
                    torch.stack([torch.sum(one_hot(torch.LongTensor(batch[3][sent_id][p]),num_classes=REFERENCE_SIZE),dim=-2) if len(batch[3][sent_id][p])>0 else torch.LongTensor([0]*REFERENCE_SIZE) for p in range(args.max_seq_length)]) \
                    for sent_id in range(len(batch[3]))])
                # logger.info(batch[0].size())
                # logger.info(batch[1].size())
                # logger.info(batch[2].size())
                # logger.info(label_type_ids.size())
                inputs = {"input_ids":batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "tag_ids":batch[2].to(args.device),
                "label_type_ids": label_type_ids.to(args.device)
                }
                # forward & backward
                outputs = model(**inputs)
                loss = outputs[0]
                # logger.info(loss)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
    return global_step, tr_loss / global_step

def load_and_cache_examples(args, tokenizer, pickle_file, masked_token_label_id):
    "Due to the sparsity of the type, it's wasteful to save the data as tensor"
    filename = pickle_file.split('/')[-1]
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}".format(filename.replace('.pickle',''),
            "uncased" if args.do_lower_case else "cased",
            str(args.max_seq_length)))
    if (os.path.exists(cached_features_file) and not args.overwrite_cache):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
    else: 
        logger.info("Creating features from dataset file %s", args.data_dir)
        features = convert_pickle_to_features(data_dir=args.data_dir, 
                                              filename=pickle_file,
                                              tokenizer=tokenizer,
                                              max_seq_length=args.max_seq_length,
                                              pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                              attention_mask_id=0,
                                              num_reference=REFERENCE_SIZE,
                                              output_ignore_id= masked_token_label_id,
                                              do_lower_case=args.do_lower_case)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
                # if args.yago_reference:
                #     torch.save(ref_features, cached_yago_file)
    logger.info(features[0].input_ids)
    logger.info(features[0].input_mask)
    logger.info(features[0].tag_ids)
    logger.info(features[0].type_ids)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_type_ids = [f.type_ids for f in features]
    assert (len(all_input_ids) == len(all_input_mask))
    assert (len(all_tag_ids) == len(all_type_ids))
    assert (len(all_input_mask) == len(all_type_ids))
    # logger.info(all_type_ids)
    # dataset = MultiNerDataset(all_input_ids, all_input_mask, all_tag_ids, all_type_ids)
    # dataset = [[all_input_ids[i], all_input_mask[i], all_tag_ids[i], all_type_ids[i]] for i in range(len(all_input_ids))]
    dataset = [all_input_ids, all_input_mask, all_tag_ids, all_type_ids]
    logger.info("len(dataset):{}".format(len(dataset)))

    return dataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the files for bert pretraining.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_REPO, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    # parser.add_argument("--config_name", default="", type=str,
    #                     help="Pretrained config name or path if not the same as model_name")
    # parser.add_argument("--tokenizer_name", default="", type=str,
    #                     help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_REPO, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    # parser.add_argument("--do_eval", action="store_true",
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_predict", action="store_true",
    #                     help="Whether to run predictions on the test set."))
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    # parser.add_argument("--eval_all_checkpoints", action="store_true",
    #                     help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    args = parser.parse_args()

    if 'uncased' in args.model_name_or_path:
        args.do_lower_case = True
    else:
        args.do_lower_case = False

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Use cross entropy ignore index (-100) as padding label id so that only real label ids contribute to the loss later
    masked_token_label_id = CrossEntropyLoss().ignore_index
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = (BertConfig, BertForPreTraining, BertTokenizer)
    
    # bertconfig = BertConfig.from_pretrained('bert-base-uncased',
    #                                 do_lower_case=args.do_lower_case,
    #                                 cache_dir='/work/smt2/qfeng/Project/huggingface/pretrain/base_uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                    do_lower_case=args.do_lower_case,
                                    cache_dir='/work/smt2/qfeng/Project/huggingface/pretrain/base_uncased')
    
    config = BertMultipleLabelConfig.from_pretrained('bert-base-uncased' if args.do_lower_case else 'bert-base-cased',
                                                     reference_size = int(REFERENCE_SIZE),
                                                     lstm_hidden_size = 200,
                                                     num_tags = 3,
                                                     cache_dir='/work/smt2/qfeng/Project/huggingface/pretrain/base_{}'.format('uncased' if args.do_lower_case else 'cased'))
    
    model = BertForMultipleLabelTokenClassification.from_pretrained(args.model_name_or_path,
                                                                    from_tf=bool(".ckpt" in args.model_name_or_path),
                                                                    config=config,
                                                                    cache_dir='/work/smt2/qfeng/Project/huggingface/pretrain/base_{}'.format('uncased' if args.do_lower_case else 'cased'))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    global_step, tr_loss = train(args, model=model, tokenizer=tokenizer, masked_token_label_id=masked_token_label_id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == "__main__":
    main()
