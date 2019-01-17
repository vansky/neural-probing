import os
import torch
import gzip

from nltk import sent_tokenize

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class SentenceCorpus(object):
    def __init__(self, path, vocab_file, classifier_vocab_file,
                 test_flag=False, train_classifier_flag=False,
                 test_classifier_flag=False, interactflag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt'):
        if not test_flag and not train_classifier_flag:
            self.dictionary = Dictionary()
            self.class_dictionary = Dictionary()
            self.train,self.train_classes = self.tokenize(os.path.join(path, trainfname))
            self.valid,self.valid_classes = self.tokenize_with_unks(os.path.join(path, validfname))
            self.vocab_file = self.save_dict(vocab_file, self.dictionary)
        elif train_classifier_flag:
            self.dictionary = Dictionary()
            self.load_dict(vocab_file, self.dictionary)
            self.class_dictionary = Dictionary()
            # The classifier might be trained on different data from the lm
            # So we need to unk the training data as well as the validation data
            self.train,self.train_classes = self.tokenize_with_unks(os.path.join(path, trainfname))
            self.valid,self.valid_classes = self.tokenize_with_unks(os.path.join(path, validfname))
            self.classifier_vocab_file = self.save_dict(classifier_vocab_file, self.class_dictionary)
        elif test_classifier_flag:
            # Test the classifier
            self.dictionary = Dictionary()
            self.load_dict(vocab_file, self.dictionary)
            self.class_dictionary = Dictionary()
            self.load_dict(classifier_vocab_file, self.class_dictionary)
            if not interactflag:
                self.test,self.test_classes = self.sent_tokenize_with_unks(os.path.join(path, testfname))
        else:
            # Test the LM
            self.dictionary = Dictionary()
            self.load_dict(vocab_file, self.dictionary)
            # Create an empty class_dictionary to reuse code
            self.class_dictionary = Dictionary()
            if not interactflag:
                self.test,self.test_classes = self.sent_tokenize_with_unks(os.path.join(path, testfname))    

    def save_dict(self, path, which_dictionary):
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'wb') as f:
                torch.save(which_dictionary, f, pickle_module=dill)
        else:
            # Assume dict is plaintext
            with open(path, 'w') as f:
                for word in which_dictionary.idx2word:
                    f.write(word+'\n')

    def load_dict(self, path, which_dictionary):
        assert os.path.exists(path)
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'rb') as f:
                fdata = torch.load(f, pickle_module=dill)
                if type(fdata) == type(()):
                    # Compatibility with old pytorch LM saving
                    which_dictionary = fdata[3]
                which_dictionary = fdata
        else:
            # Assume dict is plaintext
            with open(path, 'r') as f:
                for line in f:
                    which_dictionary.add_word(line.strip())

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            FIRST = True
            lang_line = True
            for line in f:
                if line.strip() == '':
                    # Ignore blank lines
                    continue
                if lang_line:
                    if FIRST:
                        words = ['<eos>'] + line.split() + ['<eos>']
                        FIRST = False
                        # Actually, the class_line also has a FIRST,
                        # but we're just counting line lengths and noting
                        # token types here, and we know the line length
                        # lang_line already
                    else:
                        words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
                else:
                    classes = line.split() + ['<eos>']
                    for this_class in classes:
                        self.class_dictionary.add_word(this_class)
                lang_line = not lang_line

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            class_ids = torch.LongTensor(tokens)
            token = 0
            class_token = 0
            FIRST = True
            lang_line = True
            for line in f:
                if line.strip() == '':
                    # Ignore blank lines
                    continue
                if lang_line:
                    if FIRST:
                        words = ['<eos>'] + line.split() + ['<eos>']
                    else:
                        words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1
                else:
                    if FIRST:
                        classes = ['<eos>'] + line.split() + ['<eos>']
                        FIRST = False
                    else:
                        classes = line.split() + ['<eos>']
                    for this_class in classes:
                        class_ids[class_token] = self.class_dictionary.word2idx[this_class]
                        class_token += 1
                lang_line = not lang_line
        return (ids, class_ids)

    def tokenize_with_unks(self, path):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        # Determine the length of the corpus
        with open(path, 'r') as f:
            tokens = 0
            FIRST = True
            lang_line = True
            for line in f:
                if line.strip() == '':
                        # Ignore blank lines
                        continue
                if lang_line:
                    if FIRST:
                        words = ['<eos>'] + line.split() + ['<eos>']
                        FIRST = False
                    else:
                        words = line.split() + ['<eos>']
                    tokens += len(words)
                    lang_line = False
                else:
                    # All we care about is the number of tokens,
                    # which we already got from the lang_line
                    lang_line = True

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            class_ids = torch.LongTensor(tokens)
            token = 0
            class_token = 0
            FIRST = True
            lang_line = True
            for line in f:
                if line.strip() == '':
                    # Ignore blank lines
                    continue
                if lang_line:
                    if FIRST:
                        words = ['<eos>'] + line.split() + ['<eos>']
                    else:
                        words = line.split() + ['<eos>']
                    for word in words:
                        # Convert OOV to <unk>
                        if word not in self.dictionary.word2idx:
                            ids[token] = self.dictionary.add_word("<unk>")
                        else:
                            ids[token] = self.dictionary.word2idx[word]
                        token += 1
                else:
                    if FIRST:
                        classes = ['<eos>'] + line.split() + ['<eos>']
                        FIRST = False
                    else:
                        classes = line.split() + ['<eos>']
                    for this_class in classes:
                        # Convert OOV to <unk>
                        if this_class not in self.class_dictionary.word2idx:
                            class_ids[class_token] = self.class_dictionary.add_word("<unk>")
                        else:
                            class_ids[class_token] = self.class_dictionary.word2idx[this_class]
                        class_token += 1
                lang_line = not lang_line
        return (ids, class_ids)

    def sent_tokenize_with_unks(self, path):
        """Tokenizes a text file into sentences, adding unks if needed."""
        assert os.path.exists(path)
        word_ids = []
        sents = []
        class_ids = []
        class_sents = []
        lang_line = True
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    # Ignore blank lines
                    continue
                if lang_line:
                    sents.append(line.strip())
                    words = ['<eos>'] + line.split() + ['<eos>']
                    tokens = len(words)

                    # Tokenize file content
                    ids = torch.LongTensor(tokens)
                    token = 0
                    for word in words:
                        # Convert OOV to <unk>
                        if word not in self.dictionary.word2idx:
                            ids[token] = self.dictionary.add_word("<unk>")
                        else:
                            ids[token] = self.dictionary.word2idx[word]
                        token += 1
                    word_ids.append(ids)
                else:
                    class_sents.append(line.strip())
                    classes = ['<eos>'] + line.split() + ['<eos>']
                    tokens = len(classes)

                    # Tokenize file content
                    ids = torch.LongTensor(tokens)
                    token = 0
                    for this_class in classes:
                        # Convert OOV to <unk>
                        if this_class not in self.class_dictionary.word2idx:
                            ids[token] = self.class_dictionary.add_word("<unk>")
                        else:
                            ids[token] = self.class_dictionary.word2idx[this_class]
                        token += 1
                    class_ids.append(ids)
                lang_line = not lang_line
        return ((sents, word_ids),(class_sents,class_ids))

    def online_tokenize_with_unks(self, line):
        """Tokenizes an input sentence, adding unks if needed."""
        all_ids = []
        sents = [line.strip()]

        words = ['<eos>'] + line.strip().split() + ['<eos>']
        tokens = len(words)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            # Convert OOV to <unk>
            if word not in self.dictionary.word2idx:
                ids[token] = self.dictionary.add_word("<unk>")
            else:
                ids[token] = self.dictionary.word2idx[word]
            token += 1
        all_ids.append(ids)
        return (sents, all_ids)
