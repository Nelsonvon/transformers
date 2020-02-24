import sys
import string
import datetime
import logging
from nltk.tokenize import word_tokenize

from transformers import BertTokenizer
logger = logging.getLogger(__name__)

def decode_preprocessing(dataset, output_file, tokenizer, max_len, mode):
    
    with open(dataset, 'r') as fin:
        input_lines = fin.readlines()
    starttime = datetime.datetime.now()
    if mode == 'sent_line':
        lines = []
        for line in input_lines:
            tok_line = word_tokenize(line.strip())
            lines += [token + ' O\n' for token in tok_line] + ['\n']
    else:
        lines = input_lines



    subword_len_counter = 0

    last_punc_buffer = ""
    output = ""
    for line in lines:
        line_copy = line
        line = line.rstrip()

        if not line:
            # print(line)
            output += line + '\n'
            last_punc_buffer = ""
            subword_len_counter = 0
            continue

        token = line.split()[0]

        current_subwords_len = len(tokenizer.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if all(char in string.punctuation for char in token) and line.split()[1] == 'O':
            last_punc_buffer = ""
        else:
            last_punc_buffer += line_copy


        if (subword_len_counter + current_subwords_len) > max_len:
            # print("")
            output += '\n'
            # print(last_punc_buffer.rstrip())
            output += last_punc_buffer.rstrip() + '\n'
            subword_len_counter = len(last_punc_buffer.split('\n'))
            last_punc_buffer = ""
            continue

        subword_len_counter += current_subwords_len

        # print(line)
        output += line + '\n'
    endtime = datetime.datetime.now()
    duration = (endtime-starttime).total_seconds()
    logger.info(duration)
    with open(output_file, 'w') as fout:
        fout.write(output)
    
    return