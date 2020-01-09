import pickle
import random
from typing import List, Dict
from transformers import BertTokenizer
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Warning: DO NOT change the definition of this class!
class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

class InputFeatures(object):
    """A single training instance as feature"""

    def __init__(self, input_ids, input_mask, segment_ids, output_ids, is_next):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.output_ids = output_ids
        self.is_next = is_next

class ReferenceFeatures(object):
    """Structure to store additional yago reference input"""

    def __init__(self, reference_ids, reference_weights):
        self.reference_ids = reference_ids
        self.reference_weights = reference_weights

def generate_masked_sent(tokenizer: BertTokenizer, sent: str, mask_token: str, mask_rate:float=0.15):
    words = tokenizer.tokenize(sent)
    input_list = words
    output_list = words
    len_sent = len(words)
    num_mask_token = int(len_sent*mask_rate)
    masked_idx = random.sample(list(range(len_sent), num_mask_token))
    for idx in masked_idx:
        # TODO: randomly masks / replaces / keeps the token
        input_list[idx] = mask_token
    return input_list, output_list

def examples_generator(tokenizer: BertTokenizer, corpus_pickle: str, save_examples: bool=True, save_path:str=None, random_seed: bool=False, seed: int=0):
    """
    # Load the corpus from pickle file. {guid: text}
    # generate training examples for MLM and NSP tasks.
    # return the examples, save the examples if needed.
    """
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token
    pad_token = tokenizer.pad_token
    with open(corpus_pickle, 'rb') as fin:
        sentence_dict = pickle.load(fin)
    if random_seed:
        random.seed(seed)
    MAX_SENTENCE_IDX = len(sentence_dict)-1
    for guid in sentence_dict:
        sent1:str = sentence_dict[guid]
        words_1, labels_1 = generate_masked_sent(tokenizer = tokenizer, sent = sent1, mask_token = mask_token)
        dice_IsNext = random.choice([0,1])
        if dice_IsNext == 0:
            guid2 = (guid +1)%(MAX_SENTENCE_IDX+1)
            labels.append("[IsNext]")
        else:
            guid2 = random.randint(0,MAX_SENTENCE_IDX)
            labels.append("[NotNext]")
        sent2:str = sentence_dict[guid2]
        words_2, labels_2 = generate_masked_sent(tokenizer = tokenizer, sent=sent2, mask_token = mask_token)
        words = [cls_token] + words_1 + [sep_token] + words_2 + [sep_token]
        labels = [pad_token] + labels_1 + [sep_token] + words_2 + [sep_token]

def read_examples_from_pickle(data_dir):
    test_mod = True
    test_files = 0
    examples = []
    for file in os.listdir(data_dir):
        if file.startswith("cached_"):
            continue
        if test_mod and not file.startswith('wiki'):
            continue
        with open(os.path.join(data_dir, file), 'rb') as fin:
            logger.info("reading pickle file {}".format(file))
            try:
                instances = pickle.load(fin)
                examples.extend(instances)
                if test_mod:
                    test_files += 1
            except:
                logger.info("failed to read pickle file {}".format(file))
        if test_mod and test_files == 1:
            break            
    return examples

def convert_examples_to_features(examples, 
                                tokenizer: BertTokenizer,
                                max_seq_length,
                                pad_token_id=0,
                                pad_token_segment_id=0,
                                mask_padding_with_zero=True,
                                yago_ref=False,
                                MAX_NUM_REFERENCE=10):
    features = []
    ref_features = []
    if yago_ref:
        with open('/work/smt3/wwang/TAC2019/qihui_data/yago/YagoReference.pickle', 'rb') as ref_pickle: #TODO:
            ref_dict: Dict = pickle.load(ref_pickle)

    exp_iterator = tqdm(examples, desc='Instances', total=len(examples))
    count_example = 0
    test_mod = True
    for example in exp_iterator:
        count_example += 1
        if test_mod and count_example == 1000:
            break
        masked_tokens = example.tokens
        output_tokens = type(masked_tokens)(masked_tokens)
        segment_ids = example.segment_ids

        for pos_idx in range(len(example.masked_lm_positions)):
            output_tokens[example.masked_lm_positions[pos_idx]] = example.masked_lm_labels[pos_idx]
        
        assert(tokenizer.mask_token == "[MASK]")
        assert(tokenizer.pad_token_id == pad_token_id)

        input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
        output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        assert(len(input_ids) == len(output_ids))

        input_mask = [1 if mask_padding_with_zero else 0]*len(input_ids)
        padding_length = max_seq_length - len(input_ids)

        if yago_ref:
            reference_ids = [(list(ref_dict[input_id].keys()) if (input_id in ref_dict) else [pad_token_id]) for input_id in input_ids]
            reference_weights = [(list(ref_dict[input_id].values()) if (input_id in ref_dict) else [0.0]) for input_id in input_ids]
            assert (len(reference_ids)==len(input_ids))

            max_ref = MAX_NUM_REFERENCE

            for i in range(len(reference_ids)):
                assert (len(reference_ids[i]) == len(reference_weights[i]))
                reference_ids[i] += [pad_token_id]*(max_ref-len(reference_ids[i]))
                reference_weights[i] +=  [0.0]*(max_ref-len(reference_weights[i]))
                assert (len(reference_ids[i]) == len(reference_weights[i]))
            reference_ids += [[pad_token_id]*max_ref]*padding_length
            reference_weights += [[0.0]*max_ref]*padding_length
            ref_features.append(ReferenceFeatures(reference_ids=reference_ids, reference_weights=reference_weights))
        # zero-pad up to the sequence length.
        input_ids += [pad_token_id]*padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        output_ids += [pad_token_id]*padding_length

        assert(len(input_ids)== max_seq_length)
        assert(len(input_mask)== max_seq_length)
        assert(len(segment_ids)== max_seq_length)
        assert(len(output_ids)== max_seq_length)
        features.append(InputFeatures(input_ids, input_mask, segment_ids, output_ids, 1 if example.is_random_next else 0))
        """next_sentence_label:
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

        """



    return(features, ref_features)
