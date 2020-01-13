import pickle
from typing import List, Dict
from transformers import BertTokenizer
from tqdm import tqdm, trange

# Restrict maximal number of reference types for each token
MAX_NUM_TYPES=10

if True: #__name__= '__main__'
    do_lower_case=True
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if do_lower_case else 'bert-base-cased',
                                                      do_lower_case=do_lower_case,
                                                      cache_dir='/work/smt2/qfeng/Project/huggingface/pretrain/{}'.format('base-uncased' if do_lower_case else 'base-cased'))

    type_idx_dict = {tokenizer.pad_token: tokenizer.pad_token_id}
    idx_type_dict = {tokenizer.pad_token_id: tokenizer.pad_token}
    reference_dict ={} # Dict(token_id, Dict(reference_id, count))

    print("processing starts")
    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/yagoTypes.tsv', 'r') as fin:
        fin.readline()
        count_lines = 0
        lines = tqdm(fin, desc="lines")
        for line in lines:
            elements = line.strip().split('\t')
            entity = elements[1][1:-1]
            if len(entity)>2 and entity[2] == '/':
                entity = entity[3:]
            entry_type = elements[2]
            entity_type = elements[3][1:-1]
            if entry_type != 'rdf:type' or entity_type.startswith('wikicat'):
                continue

            if entity_type not in type_idx_dict:
                new_id = len(type_idx_dict)
                assert(new_id not in idx_type_dict)
                type_idx_dict[entity_type] = new_id
                idx_type_dict[new_id] = entity_type

            if do_lower_case:
                token_ids = tokenizer.encode(entity.replace('_', ' ').lower())
            else:
                token_ids = tokenizer.encode(entity.replace('_',' '))

            for token_id in token_ids:
                if token_id == tokenizer.cls_token_id or token_id == tokenizer.sep_token_id:
                    continue
                if token_id in reference_dict:
                    if type_idx_dict[entity_type] in reference_dict[token_id]:
                        reference_dict[token_id][type_idx_dict[entity_type]] += 1
                    else:
                        reference_dict[token_id][type_idx_dict[entity_type]] = 1
                else:
                    reference_dict[token_id]= {type_idx_dict[entity_type]: 1}
            count_lines += 1
            # if count_lines == 1000:
            #     break
    # XTODO: normalize
    for token_id in reference_dict:
        norm_reference = {k: v for k, v in sorted(reference_dict[token_id].items(), key=lambda item: item[1], reverse=True)}
        # if len(norm_reference) > MAX_NUM_TYPES:
        #     norm_reference = {k: v for k, v in list(norm_reference.items())[:MAX_NUM_TYPES]}
        sum_occur = sum(list(norm_reference.values()))
        for k in norm_reference:
            norm_reference[k] = norm_reference[k] / sum_occur
        reference_dict[token_id] = norm_reference
    # debug by visualization
    visual_debug=True
    if visual_debug:
        for count, token_id in enumerate(reference_dict):
            print(tokenizer.convert_ids_to_tokens([token_id]))
            print({idx_type_dict[ent_id]: reference_dict[token_id][ent_id] for ent_id in reference_dict[token_id]})
            if count == 10:
                break

    print("# Reference vocab: {}".format(str(len(reference_dict))))
    print("# Reference types: {}".format(str(len(type_idx_dict))))


    # XTODO: save statistics


    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/YagoReference{}_unlimit.pickle'.format('' if do_lower_case else '_cased'), 'wb') as fout:
        pickle.dump(reference_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/type_idx_dicts{}_unlimit.pickle'.format('' if do_lower_case else '_cased'), 'wb') as fout:
        pickle.dump(type_idx_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/idx_type_dicts{}_unlimit.pickle'.format('' if do_lower_case else '_cased'), 'wb') as fout:
        pickle.dump(idx_type_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

