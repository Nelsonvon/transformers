
import random
import pickle
basic_folder = '/work/smt2/qfeng/Project/huggingface/datasets/'

def split_train_dev():
    train_buffer = ""
    dev_buffer = ""
    sent_buffer = ""
    with open("/work/smt2/qfeng/Project/huggingface/datasets/TAC_1517_mixed/tac_1517_mixed.txt", 'r') as fin:
        for line in fin:
            if line == '\n':
                sent_buffer += line
                if random.random() < 0.1:
                    dev_buffer += sent_buffer
                else:
                    train_buffer += sent_buffer
                sent_buffer = ""
            else:
                sent_buffer += line
    with open("/work/smt2/qfeng/Project/huggingface/datasets/TAC_1517_mixed/train.txt",'w') as fout:
        fout.write(train_buffer)
    with open("/work/smt2/qfeng/Project/huggingface/datasets/TAC_1517_mixed/dev.txt", 'w') as fout:
        fout.write(dev_buffer)

# split_train_dev()

def split_dev_test():
    test_buffer = ""
    dev_buffer = ""
    sent_buffer = ""
    with open("/work/smt2/qfeng/Project/huggingface/datasets/TAC/test.txt", 'r') as fin:
        for line in fin:
            if line == '\n':
                sent_buffer += line
                if random.random() < 0.5:
                    dev_buffer += sent_buffer
                else:
                    test_buffer += sent_buffer
                sent_buffer = ""
            else:
                sent_buffer += line
    with open("/work/smt2/qfeng/Project/huggingface/datasets/TAC_1517_dev19/test.txt", 'w') as fout:
        fout.write(test_buffer)
    with open("/work/smt2/qfeng/Project/huggingface/datasets/TAC_1517_dev19/dev.txt", 'w') as fout:
        fout.write(dev_buffer)

# split_dev_test()

def repair_wiki():
    from qihui.data_processing.pretraining_instances_generator import repair_wiki_pickle
    repair_wiki_pickle()
    return

# repair_wiki()

def pseudo_yago_reference():
    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/YagoReference_cased.pickle', 'rb') as fin:
        reference_dict = pickle.load(fin)
    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/type_idx_dicts_cased.pickle', 'rb') as fin:
        type_idx_dict = pickle.load(fin)
    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/idx_type_dicts_cased.pickle', 'rb') as fin:
        idx_type_dict = pickle.load(fin)

    eps=1e-6
    for id in reference_dict:
        rand_type_idx = random.sample(list(idx_type_dict.keys()),k=10)
        rand_list = []
        pseudo_weights = {}
        for _ in range(9):
            rand_list.append(random.random())
        rand_list.sort(reverse=True)
        rand_list = [1.0] + rand_list + [0.0]
        pointer = 0
        for i in range(10):
            if rand_list[i]-rand_list[i+1] > eps:
                pseudo_weights[rand_type_idx[i]] = rand_list[i]-rand_list[i+1]
            else:
                rand_list[i+1] = rand_list[i]
        reference_dict[id] = pseudo_weights
    
    example_ids = random.sample(list(reference_dict.keys()), k=10)
    for id in example_ids:
        print(reference_dict[id])
        assert(abs(sum(reference_dict[id].values())-1)<eps)
    with open('/work/smt3/wwang/TAC2019/qihui_data/yago/PseudoYagoReference_cased.pickle', 'wb') as fout:
        pickle.dump(reference_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

pseudo_yago_reference()

    