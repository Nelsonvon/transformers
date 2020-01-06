import sys
import string

from transformers import BertTokenizer

dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])
cache_dir = sys.argv[4]

subword_len_counter = 0

tokenizer = BertTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)

last_punc_buffer = ""

with open(dataset, "rt") as f_p:
    for line in f_p:
        line_copy = line
        line = line.rstrip()

        if not line:
            print(line)
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
            print("")
            print(last_punc_buffer.rstrip())
            subword_len_counter = len(last_punc_buffer.split('\n'))
            last_punc_buffer = ""
            continue

        subword_len_counter += current_subwords_len

        print(line)