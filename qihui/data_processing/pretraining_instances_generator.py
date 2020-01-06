import random
import datetime
import collections
import logging
from transformers import BertTokenizer
import os
import pickle
from qihui.model.utils_bert import TrainingInstance
import argparse
logger = logging.getLogger(__name__)

# class TrainingInstance(object):
#   """A single training instance (sentence pair)."""

#   def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
#                is_random_next):
#     self.tokens = tokensut
#     self.segment_ids = segment_ids
#     self.is_random_next = is_random_next
#     self.masked_lm_positions = masked_lm_positions
#     self.masked_lm_labels = masked_lm_labels

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def create_training_instances(input_files, tokenizer: BertTokenizer, rng, max_seq_length=128,
                              dupe_factor=2, short_seq_prob=0.1, masked_lm_prob=0.15,
                              max_predictions_per_seq=20, corpus='wiki'):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  # (3) For Bookcorpus, since the corpus we have has no blank lines, we split
  # corpus into documents with maximal 200 sentences. Instances are processed 
  # for every 10K and generates a pickle file
  MAX_LINES_PER_DOC_IN_BOOK = 200
  MAX_DOCS_PER_FILE = 10000
  file_suffix = 0
  docs_in_file = 0
  vocab_words = list(tokenizer.vocab.keys())
  for input_file in input_files:
    count_line = 0
    lines_in_doc = 0
    with open(input_file, "r") as reader:
      while True:
        line = reader.readline()
        if not line:
          break
        line = line.strip()

        if corpus == 'book':
          lines_in_doc += 1
        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
          docs_in_file +=1
          lines_in_doc = 0
        elif corpus == 'book' and lines_in_doc == MAX_LINES_PER_DOC_IN_BOOK:
          all_documents.append([])
          docs_in_file +=1
          lines_in_doc = 0
        tokens = tokenizer.tokenize(line.lower()) # bookcorpus is uncased, thus we need to lowercase all sentences.
        if tokens:
          all_documents[-1].append(tokens)
        count_line +=1
        if (count_line+1)%100000 == 0:
          logger.info(datetime.datetime.now())
          count_line = 0
        if corpus == 'book' and docs_in_file >= MAX_DOCS_PER_FILE:
          all_documents = [x for x in all_documents if x]
          rng.shuffle(all_documents)          
          instances = []
          for _ in range(dupe_factor):
            for document_index in range(len(all_documents)):
              instances.extend(
                  create_instances_from_document(
                      all_documents, document_index, max_seq_length, short_seq_prob,
                      masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

          rng.shuffle(instances)
          with open(os.path.join(instances_dir, 'book{}_{}.pickle'.format(args.file_suffix, str(file_suffix))),'wb') as fout:
            pickle.dump(instances, fout, protocol=pickle.HIGHEST_PROTOCOL)
            print('finished processing book_{}.pickle'.format(args.file_suffix))
          all_documents = [[]]
          docs_in_file = 0
          lines_in_doc = 0
          file_suffix += 1

  if not corpus == 'book':
    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    # vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
      for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_length, short_seq_prob,
                masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances
  else:
    return None


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.

    # if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
    #     token.startswith("##")):
    #   cand_indexes[-1].append(i)
    # else:
    cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def repair_wiki_pickle():
  data_dir = '/work/smt3/wwang/TAC2019/qihui_data/bert_train_instances/'
  wiki_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/wikiextractor/text/'
  raw_sent_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/raw_tk_sent/'
  instances_dir = '/work/smt3/wwang/TAC2019/qihui_data/bert_train_instances/'
  rng = random.Random(0)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                            do_lower_case=True,
                                            cache_dir='/work/smt2/qfeng/Project/huggingface/pretrain/base_uncased')
  for file in os.listdir(data_dir):
    if not file.startswith("wiki_"):
      continue
    with open(os.path.join(data_dir, file), 'rb') as fin:
      try:
        instances = pickle.load(fin)
        logger.info("Successfully read pickle file {}".format(file))
      except:
        logger.info("failed to read pickle file {}".format(file))
        suffix = file[len('wiki_'):len('wiki_')+2]
        input_files = [os.path.join(raw_sent_dir, 'wiki_raw_{}'.format(suffix))]
        instances = create_training_instances(input_files, tokenizer, rng)
        with open(os.path.join(instances_dir, 'wiki_{}.pickle'.format(suffix)),'wb') as fout:
          pickle.dump(instances, fout, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('finished repairing wiki_{}.pickle'.format(suffix))


if __name__ == "__main__":  
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_suffix', default='A')
  parser.add_argument('--corpus') # wiki / book
  args = parser.parse_args()

  rng = random.Random(0)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                  do_lower_case=True,
                                                  cache_dir='/work/smt2/qfeng/Project/huggingface/pretrain/base_uncased')
  # input_files = ['/work/smt3/wwang/TAC2019/qihui_data/wiki/raw_tk_sent/wiki_raw_CA']

  print("generation starts")

  if args.corpus == 'wiki':
    wiki_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/wikiextractor/text/'
    raw_sent_dir = '/work/smt3/wwang/TAC2019/qihui_data/wiki/raw_tk_sent/'
    instances_dir = '/work/smt3/wwang/TAC2019/qihui_data/bert_train_instances/'

    for suffix in os.listdir(wiki_dir):
      if os.path.exists(os.path.join(instances_dir, 'wiki_{}.pickle'.format(suffix))):
        continue
      if not suffix.startswith(args.file_suffix):
        continue
      # if suffix == 'AA': # will remove later, now guarantee only the well-processed corpora are used for generating instances.
      #   break
      input_files = [os.path.join(raw_sent_dir, 'wiki_raw_{}'.format(suffix))]
      instances = create_training_instances(input_files, tokenizer, rng)
      with open(os.path.join(instances_dir, 'wiki_{}.pickle'.format(suffix)),'wb') as fout:
        pickle.dump(instances, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print('finished processing wiki_{}.pickle'.format(suffix))
  else:
    raw_sent_dir = '/work/smt3/wwang/TAC2019/qihui_data/bookcorpus/bookcorpus/'
    instances_dir = '/work/smt3/wwang/TAC2019/qihui_data/bert_train_instances/'
    input_files = [os.path.join(raw_sent_dir, 'books_large_p{}.txt'.format(args.file_suffix))]
    instances = create_training_instances(input_files, tokenizer, rng, corpus=args.corpus)
    if instances:
      with open(os.path.join(instances_dir, 'book_{}.pickle'.format(args.file_suffix)),'wb') as fout:
        pickle.dump(instances, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print('finished processing book_{}.pickle'.format(args.file_suffix))