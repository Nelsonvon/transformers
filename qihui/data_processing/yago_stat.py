
from tqdm import tqdm
ent_dict = {}
wordnet_dict = {}
alltag_dict = {}
with open('/work/smt3/wwang/TAC2019/qihui_data/yago/yagoTypes.tsv', 'r') as fin:
    lines = tqdm(fin,desc='lines')
    for line in lines:
        entity = line.split('\t')[1]
        tag = line.split('\t')[3]
        if entity not in ent_dict:
            ent_dict[entity] = 1
        if (not tag.startswith('<wikicat_'))  and tag not in wordnet_dict:
            wordnet_dict[tag] = 1
        if tag not in alltag_dict:
            alltag_dict[tag] = 1
    print(len(ent_dict))
    print(len(wordnet_dict))
    print(len(alltag_dict))