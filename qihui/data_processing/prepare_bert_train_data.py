from nltk.tokenize import sent_tokenize, word_tokenize
import json
import os

def wiki_get_raw_sentences():
    wiki_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/wikiextractor/text/'
    output_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/raw_tk_sent'
    
    for sub_dir in os.listdir(wiki_dir):
        if os.path.exists(os.path.join(output_dir, 'wiki_raw_{}'.format(sub_dir))):
            continue
        fout = open(os.path.join(output_dir, 'wiki_raw_{}'.format(sub_dir)), 'w')
        for wiki_file in os.listdir(os.path.join(wiki_dir,sub_dir)):
            with open(os.path.join(wiki_dir, sub_dir, wiki_file), 'r') as fin:         
                buffer = ""
                # for line in fin:
                while(True):
                    line = None
                    try:
                        line = fin.readline()
                        if not line:
                            break
                        text: str = json.loads(line)['text']
                        # fout.write('\n'.join(sent_tokenize(text.replace('\n',' '))))
                        buffer += '\n'.join([' '.join(word_tokenize(sent)) for sent in sent_tokenize(text)]) + '\n\n'
                    except:
                        print('error in {}, break'.format(sub_dir))
                        break
                fout.write(buffer)
                # buffer = ""
        print('finished directory {}'.format(sub_dir))

def test_run():
    wiki_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/wikiextractor/text/'
    output_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/raw_sentences'
    sub_dir = 'AA'
    for wiki_file in os.listdir(os.path.join(wiki_dir,sub_dir)):
        with open(os.path.join(wiki_dir, sub_dir, wiki_file), 'r') as fin:
            lines = fin.readlines()
        with open(os.path.join(output_dir, sub_dir + '_' + wiki_file),'w') as fout:
            for line in lines:
                text: str = json.loads(line)['text']
                fout.write('\n'.join([' '.join(word_tokenize(sent)) for sent in sent_tokenize(text)]) + '\n')
    print('finished directory {}'.format(sub_dir))

wiki_get_raw_sentences()
# test_run()

