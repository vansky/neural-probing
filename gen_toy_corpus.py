''' Generates a small corpus based on a toy PCFG.

The corpus has NUM_SENTS items with the proportions given in the PCFG.
The lexical items are sampled from the lexicon and look like Noun_3.
'''

import math
import random

NUM_SENTS = 250000
FILTER = True
RM_RV_SUBJECT = False
RM_RV_OBJECT = True

Lexicon = {'DP': ('Det',3),
           'JJ': ('Adj',100),
           'NN': ('Noun',1000),
           'IV': ('LexIV',1000),
           'RV': ('LexIV',200),
           'TV': ('LexTV',1000),
           'SV': ('LexSV',100),
           'C':('Comp',2),
           'REL':('Rel',2),
           'PREP':('Prep',8),
           'WAS':('was',1)}

# This grammar has no lexical ambiguity, but has a 70-30 skew for CS
Grammar_noambig = {'S': [('NP VP',1.0)],
                   'SubS': [('NP VP-noC',1.0)],
                   'NP': [('DP NP-aDP',0.5),('NP-aDP',0.5)],
                   'NP-aDP': [('JJ NN',0.5),('NN',0.5)],
                   'VP': [('IV',1.0/3),('TV NP',1.0/3),('SV CS',1.0/3)],
                   'VP-noC': [('IV',0.5),('TV NP',0.5)],
                   'CS': [('C SubS',0.7),('SubS',0.3)]}

# This grammar has lexical ambiguity w/ RRCs and a 75-25 skew to ORCs
Grammar_ambig = {'S': [('NP VP',1.0)],
                 'NP': [('DP NP-aDP',0.5),('NP-aDP',0.5)],
                 'NP-aDP': [('JJ NN',0.25),('NN',0.25),('JJ NN-bRC',0.25),('NN-bRC',0.25)],
                 'NP-noRC': [('DP JJ NN',0.25),('DP NN',0.25),('JJ NN',0.25),('NN',0.25)],
                 'VP': [('IV',0.25),('IV PREP NP',0.25),('TV NP',0.25),('TV NP PREP NP',0.25)],
                 'VP-noC-noRC': [('RV',0.5),('RV PREP NP-noRC',0.5)],
                 'NN-bRC': [('NN REL WAS VP-noC-noRC',0.75),('NN VP-noC-noRC',0.25)]}
#                 'SubS': [('NP VP-noC',1.0)],
#                 'VP-noC': [('IV',0.25),('IV PREP NP',0.25),('TV NP',0.25),('TV NP PREP NP',0.25)],
#                 'CS': [('C SubS',0.7),('SubS',0.3)]}

Grammar_toy = {'S': [('NP VP',1.0)],
               'NP': [('JJ NN',0.5),('NN',0.5)],
               'VP': [('IV',0.5),('TV NP',0.5)]}

Grammar = Grammar_ambig

explore = [[1.0,'*','S']]
finished_trees = []
while explore != []:
    this_sent = explore.pop()
    this_prob = this_sent[0]
    last_index = len(this_sent) - 2
    NEXT = False
    prev_words = [this_prob]
    for word_index,word in enumerate(this_sent[1:]):
        if word == '*':
            NEXT = True
            continue
        elif NEXT:
            # This is the word we need to expand next
            if word in Grammar:
                # we're at a non-terminal
                for expansion in Grammar[word]:
                    # Need to used word_index+2 because the prob adds 1
                    # and the current word adds 1
                    explore.append([this_prob*expansion[1]] + \
                                   prev_words[1:] + ['*'] + expansion[0].split() + \
                                   this_sent[word_index+2:])
            else:
                # we're at a terminal
                if word_index == last_index:
                    # we're done with all the non-terminals in this tree
                    finished_trees.append(prev_words + [word])
                else:
                    # there are still non-terminals
                    explore.append(prev_words + [word] + ['*'] + this_sent[word_index+2:])
            # Expanded the current word, start again
            # TODO: This may not be the most efficient way
            break
        else:
            # not at the next breakpoint, nor the next word
            prev_words.append(word)
            
for tree in finished_trees:
    if FILTER:
        PREVERB = True
        SKIP = False
        if RM_RV_SUBJECT:
            # Remove trees with RVs in subject
            for word in tree[1:]:
                if PREVERB:
                    if word in ('IV','TV'):
                        PREVERB = False
                    elif word == 'RV':
                        # RV in subject
                        SKIP = True
                        break
                else:
                    break
        if RM_RV_OBJECT:
            # Remove trees with RVs in Object
            for word in tree[1:]:
                if PREVERB:
                    if word in ('IV','TV'):
                        PREVERB = False
                else:
                    if word == 'RV':
                        # RV in object
                        SKIP = True
                        break
        if SKIP:
            # Don't output this sentence
            continue
    for tree_i in range(int(NUM_SENTS * tree[0])):
        # Generate the right number of variants of this tree
        # Sample from the possible vocab items to make that happen
        output = []
        for word in tree[1:]:
            output.append(Lexicon[word][0]+'_'+str(random.randint(0,Lexicon[word][1])))
        print(' '.join(output))
        print(' '.join(tree[1:]))
