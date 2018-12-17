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

class Linguistic_Corpus(object):
    def __init__(self, path, vocab_file, testflag=False, interactflag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt'):
        if vocab_file[-3:] == 'bin':
            self.load_dict(vocab_file)
        else:
            self.dictionary = Dictionary()
            self.load_dict(vocab_file)
        if not testflag:
            self.dictionary = Dictionary()
            self.train = self.tokenize_with_unks(os.path.join(path, trainfname))
            self.valid = self.tokenize_with_unks(os.path.join(path, validfname))
        else:
            if not interactflag:
                self.test = self.sent_tokenize_with_unks(os.path.join(path, testfname))

    def load_dict(self, path):
        assert os.path.exists(path)
        if path[-3:] == 'bin':
            #this check actually seems to be faster than passing in a binary flag
            #assume binarized
            import dill
            with open(path, 'rb') as f:
                fdata = torch.load(f, pickle_module=dill)
                if type(fdata) == type(()):
                    # compatibility with old pytorch LM saving
                    self.dictionary = fdata[3]
                self.dictionary = fdata
        else:
            #assume plaintext
            with open(path, 'r') as f:
                for line in f:
                    self.dictionary.add_word(line.strip())

    def tokenize_with_unks(self, path):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        if path[-2:] == 'gz':
            # Determine the length of the corpus
            with gzip.open(path, 'rb') as f:
                tokens = 0
                FIRST = True
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with gzip.open(path, 'rb') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                FIRST = True
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
        else:
            # Determine the length of the corpus
            with open(path, 'r') as f:
                tokens = 0
                FIRST = True
                for fchunk in f:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                FIRST = True
                for fchunk in f:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
        return ids

    def sent_tokenize_with_unks(self, path):
        """Tokenizes a text file into sentences, adding unks if needed."""
        assert os.path.exists(path)
        all_ids = []
        sents = []
        if path [-2:] == 'gz':
            with gzip.open(path, 'rb') as f:
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        tokens = len(words)

                        # tokenize file content
                        ids = torch.LongTensor(tokens)
                        token = 0
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
                        all_ids.append(ids)
        else:
            with open(path, 'r') as f:
                for fchunk in f:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        tokens = len(words)

                        # tokenize file content
                        ids = torch.LongTensor(tokens)
                        token = 0
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
                        all_ids.append(ids)
        return (sents, all_ids)

    def online_tokenize_with_unks(self, line):
        """Tokenizes an input sentence, adding unks if needed."""
        all_ids = []
        sents = [line.strip()]

        words = ['<eos>'] + line.strip().split() + ['<eos>']
        tokens = len(words)

        # tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            # unk words that are OOV
            if word not in self.dictionary.word2idx:
                ids[token] = self.dictionary.add_word("<unk>")
            else:
                ids[token] = self.dictionary.word2idx[word]
            token += 1
        all_ids.append(ids)
        return (sents, all_ids)

class Probing_Corpus(object):
    def __init__(self, path, vocab_file, testflag=False, interactflag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt'):
        if not testflag:
            self.dictionary = Dictionary()
            self.train = self.tokenize_with_unks(os.path.join(path, trainfname))
            self.valid = self.tokenize_with_unks(os.path.join(path, validfname))
            self.vocab_file = self.save_dict(vocab_file)
        else:
            if vocab_file[-3:] == 'bin':
                self.load_dict(vocab_file)
            else:
                self.dictionary = Dictionary()
                self.load_dict(vocab_file)
            if not interactflag:
                self.test = self.sent_tokenize_with_unks(os.path.join(path, testfname))

    def save_dict(self, path):
        if path[-3:] == 'bin':
            #this check actually seems to be faster than passing in a binary flag
            #assume binarized
            import dill
            with open(path, 'wb') as f:
                torch.save(self.dictionary, f, pickle_module=dill)
        else:
            #assume plaintext
            with open(path, 'w') as f:
                for word in self.dictionary.idx2word:
                    f.write(word+'\n')

    def load_dict(self, path):
        assert os.path.exists(path)
        if path[-3:] == 'bin':
            #this check actually seems to be faster than passing in a binary flag
            #assume binarized
            import dill
            with open(path, 'rb') as f:
                fdata = torch.load(f, pickle_module=dill)
                if type(fdata) == type(()):
                    # compatibility with old pytorch LM saving
                    self.dictionary = fdata[3]
                self.dictionary = fdata
        else:
            #assume plaintext
            with open(path, 'r') as f:
                for line in f:
                    self.dictionary.add_word(line.strip())

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb') as f:
                tokens = 0
                FIRST = True
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            #ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        for word in words:
                            self.dictionary.add_word(word)

            # Tokenize file content
            with gzip.open(path, 'rb') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                FIRST = True
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            #ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        for word in words:
                            ids[token] = self.dictionary.word2idx[word]
                            token += 1
        else:
            with open(path, 'r') as f:
                tokens = 0
                FIRST = True
                for fchunk in f:
                    #print fchunk
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            #ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        for word in words:
                            self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                FIRST = True
                for fchunk in f:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            #ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        for word in words:
                            ids[token] = self.dictionary.word2idx[word]
                            token += 1
        return ids

    def tokenize_with_unks(self, path):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        if path[-2:] == 'gz':
            # Determine the length of the corpus
            with gzip.open(path, 'rb') as f:
                tokens = 0
                FIRST = True
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with gzip.open(path, 'rb') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                FIRST = True
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
        else:
            # Determine the length of the corpus
            with open(path, 'r') as f:
                tokens = 0
                FIRST = True
                for fchunk in f:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                FIRST = True
                for fchunk in f:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        if FIRST:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            FIRST = False
                        else:
                            words = line.split() + ['<eos>']
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
        return ids

    def sent_tokenize_with_unks(self, path):
        """Tokenizes a text file into sentences, adding unks if needed."""
        assert os.path.exists(path)
        all_ids = []
        sents = []
        if path [-2:] == 'gz':
            with gzip.open(path, 'rb') as f:
                for fchunk in f.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        tokens = len(words)

                        # tokenize file content
                        ids = torch.LongTensor(tokens)
                        token = 0
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
                        all_ids.append(ids)
        else:
            with open(path, 'r') as f:
                for fchunk in f:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        tokens = len(words)

                        # tokenize file content
                        ids = torch.LongTensor(tokens)
                        token = 0
                        for word in words:
                            # unk words that are OOV
                            if word not in self.dictionary.word2idx:
                                ids[token] = self.dictionary.add_word("<unk>")
                            else:
                                ids[token] = self.dictionary.word2idx[word]
                            token += 1
                        all_ids.append(ids)
        return (sents, all_ids)

    def online_tokenize_with_unks(self, line):
        """Tokenizes an input sentence, adding unks if needed."""
        all_ids = []
        sents = [line.strip()]

        words = ['<eos>'] + line.strip().split() + ['<eos>']
        tokens = len(words)

        # tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            # unk words that are OOV
            if word not in self.dictionary.word2idx:
                ids[token] = self.dictionary.add_word("<unk>")
            else:
                ids[token] = self.dictionary.word2idx[word]
            token += 1
        all_ids.append(ids)
        return (sents, all_ids)
