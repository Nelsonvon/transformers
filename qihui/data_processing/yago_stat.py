

ent_dict = {}
wordnet_dict = {}
with open('/work/smt3/wwang/TAC2019/qihui_data/yago/yagoTypes.tsv', 'r') as fin:
    for line in fin:
        entity = line.split('\t')[1]
        tag = line.split('\t')[3]
        if entity not in ent_dict:
            ent_dict[entity] = 1
        if tag.startswith('<wordnet_') and entity not in wordnet_dict:
            wordnet_dict[entity] = 1
    print(len(ent_dict))
    print(len(wordnet_dict))