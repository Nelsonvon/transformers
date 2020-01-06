
import random
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

repair_wiki()