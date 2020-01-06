import sys

from transformers import BertTokenizer


def stat(dataset,model_name_or_path,max_len):

    subword_len_counter = 0
    word_len_counter = 0

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    num_sent = 0
    num_longsent = 0
    num_longwp = 0
    num_s_char = 0
    num_s_char_ent = 0
    break_ent = 0
    sent_buffer = ""
    special_char_dict = {}
    with open(dataset, "rt") as f_p:
        long_sent = False
        long_wp = False
        special_char = False
        for line in f_p:
            line_copy = str(line)
            sent_buffer += line_copy
            line = line.rstrip()

            if not line:
                # print(line)
                if long_sent:
                    num_longsent +=1
                if long_wp:
                    num_longwp += 1
                if special_char:
                    print(sent_buffer + '##################')
                num_sent +=1
                subword_len_counter = 0
                word_len_counter = 0
                sent_buffer = ""
                long_wp = False
                long_sent = False
                special_char = False
                continue

            token = line.split()[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                num_s_char += 1
                if line.split()[1] == 'O':
                    num_s_char_ent += 1
                sent_buffer += '********************************\n'
                if line_copy.split()[0] not in special_char_dict:
                    special_char_dict[line_copy.split()[0]] = 1
                else:
                    special_char_dict[line_copy.split()[0]] += 1
                # print(line_copy)
                special_char = True
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                # print("")
                # print(line)
                if line.split()[1] != 'O':
                    break_ent += 1
                # subword_len_counter = 0
                long_wp = True
                # continue

            if word_len_counter > max_len:
                long_sent = True

            subword_len_counter += current_subwords_len
            word_len_counter += 1

            # print(line)
    print(num_sent)
    print(num_longsent)
    print(num_longwp)
    print(num_s_char)
    print(num_s_char_ent)
    print(break_ent)
    print(special_char_dict)

stat('/work/smt2/qfeng/Project/huggingface/datasets/TAC15/train.txt.tmp','bert-base-cased', 128)
