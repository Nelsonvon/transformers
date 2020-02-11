import os,sys
import pickle
import random
import logging
from typing import Dict, List
from tqdm import tqdm
import torch
import torch.utils.data
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np


logger = logging.getLogger(__name__)

# MAX_SENTENCE_LENGTH = 128
# MAX_REFERENCE_NUM = 20
TAG_REF = {'I':0, 'O':1, 'B':2}
NOTYPE_ID = 7303 # artificial type for 'O' tag tokens
# IGNORED_ID = -1


class MultiBertnerTrainingFeatures(object):
    """
        **input_ids**: List(seq_length)
        **tag_ids**: List(seq_length)
        **type_ids**: List(List(num_types))
    """
    def __init__(self, input_ids, input_mask, tag_ids, type_ids):
        # super().__init__()
        self.input_ids = input_ids
        self.input_mask = input_mask
        # self.output_mask = output_mask
        self.tag_ids = tag_ids
        self.type_ids = type_ids


class MultiNerDataset(Dataset):
    def __init__(self, all_input_ids, all_input_mask, all_tag_ids, all_type_ids):
        super(MultiNerDataset, self).__init__()
        assert(torch.is_tensor(all_input_ids) and torch.is_tensor(all_input_mask) and torch.is_tensor(all_tag_ids) and isinstance(all_type_ids, list))

        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        # self.all_output_mask = all_output_mask
        self.all_tag_ids = all_tag_ids
        self.all_type_ids = all_type_ids
    
    def __getitem__(self,idx):
        # if idx.is_tensor():
        #     idx_list = idx.tolist()
        
        # sample = []
        # sample.append(self.all_input_ids[idx,:])
        # sample.append(self.all_input_mask[idx,:])
        # # sample.append(self.all_output_mask[idx,:])
        # sample.append(self.all_tag_ids[idx,:])
        # if idx.is_tensor():
        #     sample.append([self.all_type_ids[sample_id] for sample_id in idx_list])
        # else:
        #     sample.append(self.all_type_ids[idx])
        #
        # return sample
        if isinstance(idx, int):
            return (self.all_input_ids[idx,:],
                    self.all_input_mask[idx,:],
                    self.all_tag_ids[idx,:],
                    self.all_type_ids[idx])
        else:
            idx_list = idx.tolist()
            return (self.all_input_ids[idx, :],
                    self.all_input_mask[idx, :],
                    self.all_tag_ids[idx, :],
                    [self.all_type_ids[sample_id] for sample_id in idx_list])
    
    def __len__(self):
        return (list(self.all_input_ids.size())[0])
        # return len(self.all_type_ids[0])
        # return 128


def tag_and_type_assignment(text_tok, tags, max_ref_num, o_type_id):
    """
        assign tags and types to the first token of the corresponding word
        output:
            tag_pos: List() position(token index) where the tags locate in (including 'O' tag)
            mask_pos: List() positions of the subsequent tokens
            tag_list: List() sequence of 'iob' tag
            type_list:List(List()) list of types for each labeled token.

            e.g. tag_pos[p] is the index of the p-th iob labeled token (which can be used in tag_list and type_list)
    """
    tag_pos = [i for i in range(len(text_tok)) if not text_tok[i].startswith('##')]
    mask_pos = [i for i in range(len(text_tok)) if text_tok[i].startswith('##')]
    tag_dict = {}
    type_dict = {}
    for span in tags:
        span_begin = int(span.split('#')[0])
        span_end = int(span.split('#')[1])

        tag_dict[span_begin] = TAG_REF['B']
        span_tag_list = tags[span]
        type_dict[span_begin] = span_tag_list if (max_ref_num is None or len(span_tag_list)<=max_ref_num) else random.sample(span_tag_list, max_ref_num)
        if span_end > span_begin:
            for p in range(span_begin+1, span_end+1):
                tag_dict[p] = TAG_REF['I']
                # span_tag_list = tags[span]
                type_dict[p] = span_tag_list if (max_ref_num is None or len(span_tag_list)<=max_ref_num) else random.sample(span_tag_list, max_ref_num)
    tag_list = [tag_dict[p] if p in tag_dict else TAG_REF['O'] for p in range(len(tag_pos))]
    # tag_list = [tag_dict[k] for k in sorted(tag_dict.keys())]
    # type_dict = [type_dict[k] for k in sorted(type_dict.keys())]
    type_list = [type_dict[p] if p in type_dict else [o_type_id] for p in range(len(tag_pos))]
    # logger.info(text_tok)
    # logger.info(tag_pos)
    # logger.info(tag_list)
    assert(len(tag_pos)==len(tag_list))
    assert(len(tag_pos)+len(mask_pos)==len(text_tok))
    return tag_pos, mask_pos, tag_list, type_list

def generate_tag_type_ids(text_tok, tag_pos, tag_list, type_list, output_ignore_id):
    """
        function for STEP 4
        ignored index: -1
    """
    tag_ids = [output_ignore_id]*len(text_tok)
    type_ids = [[]]*len(text_tok)
    for i in range(len(tag_pos)):
        tag_ids[tag_pos[i]] = tag_list[i]
        type_ids[tag_pos[i]] = type_list[i]
    return tag_ids, type_ids

# def masking_and_length_regularize(input_ids, mask_pos, tag_ids, type_ids, tokenizer: BertTokenizer):
#     # TODO: add [cls] and [sep] tokens, be aware of the shift of masked position
#     output_mask = [0 if i in mask_pos else 1 for i in range(len(input_ids))]
#     input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
#     output_mask = [0] + output_mask + [0]
#     input_mask = [1]*len(input_ids)
#     padding_length =
#     return input_ids, input_mask, output_mask, tag_ids, type_ids

def convert_pickle_to_features(data_dir=None, 
                               filename=None, 
                               tokenizer: BertTokenizer=None,
                               max_seq_length=128,
                               pad_token_id=0,
                               attention_mask_id=0,
                               output_ignore_id = -100,
                               num_reference=7303,
                               max_ref_num=None,
                               do_lower_case=True):

    """
    Input:
        pad_token_id: index represents padding in input_ids
        attention_mask_id: index for attention masking
        output_ignore_id: id which will be ignored by computing the loss. (pad token / subsequent word)
    """
    """
        main function of this script.
        STEPS:
        1. load text and tags from pickle file
        2. tokenize text,
           assign tags and type index(artificial type: NoType, assign a special index),
           record (output) masked tokens.
        3. convert token sequence to [input_ids]
        4. generate [tag_ids] and [type_ids] (using tag_pos, tag_list and type_list, other position will be labeled with an ignored index)
        5. generate [input_mask] (padding) and [output_mask] (padding + subsequent token)
        6. length regularization (fixed length of output)

        restriction on max. length (sequence of wordpieces)
        se_dict: Dict(text: Dict(span: List(ent_types)))
        Last reference is an artificial one for O-tag tokens
    """
    # MAX_REFERENCE_NUM = max_ref_num if max_ref_num is not None
    # MAX_SENTENCE_LENGTH = max_seq_length if max_seq_length is not None
    # IGNORED_ID = output_ignore_id if output_ignore_id is not None

    features = []
    count_abandon_num = 0
    is_first = True

    is_test_mod = True
    test_run = 0

    # STEP 1
    with open(os.path.join(data_dir, filename), 'rb') as fin:
        se_dict:Dict = pickle.load(fin)
    for text, tags in tqdm(se_dict.items(),desc="Sentences"):
        # logger.info(text)
        # logger.info(tags)
        if '##' in text:
            # Avoid conflicts brought by the subsequent prefix '##
            continue
        # STEP 2
        text_tok = tokenizer.tokenize(text.lower() if do_lower_case else text)
        """
            For now, long sentences are discarded.
            If the portion of these sentences is high (>5% for example), then we will try something else.
        """
        if len(text_tok)>=(max_seq_length-2): # -2 is for position of [cls] and [sep]
            count_abandon_num +=1
            continue
        tag_pos, mask_pos, tag_list, type_list = tag_and_type_assignment(text_tok, tags,
                                                                         max_ref_num=max_ref_num,
                                                                         o_type_id=(num_reference-1))

        # STEP 3
        input_ids = tokenizer.convert_tokens_to_ids(text_tok)
        
        # STEP 4
        tag_ids, type_ids = generate_tag_type_ids(text_tok, tag_pos, tag_list, type_list, output_ignore_id)

        # STEP 5
        # output_mask = [mask_id if i in mask_pos else 1-mask_id for i in range(len(input_ids))]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        # output_mask = [mask_id] + output_mask + [mask_id]
        input_mask = [1- attention_mask_id]*len(input_ids)
        padding_length = max_seq_length - len(input_ids)

        # STEP 6 
        # add [cls] and [sep] tokens, be aware of the shift of masked position
        input_ids += [pad_token_id]*padding_length
        input_mask += [attention_mask_id]*padding_length
        # output_mask += [attention_mask_id]*padding_length
        tag_ids = [output_ignore_id] + tag_ids + [output_ignore_id]*(padding_length+1)
        type_ids = [[]] + type_ids + [[]]*(padding_length+1)

        assert(len(input_ids)== max_seq_length)
        assert(len(input_mask)== max_seq_length)
        # assert(len(output_mask)== max_seq_length)
        assert(len(tag_ids)== max_seq_length)
        assert(len(type_ids)== max_seq_length)
        features.append(MultiBertnerTrainingFeatures(input_ids, input_mask, tag_ids, type_ids))
        if is_first:
            logger.info(text_tok)
            logger.info(input_ids)
            logger.info(input_mask)
            # logger.info(output_mask)
            logger.info(tag_ids)
            logger.info(type_ids)
            is_first = False
        test_run +=1
        if test_run == 1000 and is_test_mod:
            break
    return features