import sys
import random
import numpy as np

lex = []
tree = []
filler_lex = []
filler_trees = []
optiona_lex = []
optiona_trees = []
optionb_lex = []
optionb_trees = []

proportion = 3 # how many times more items in the larger than the smaller

for line in sys.stdin:
    if lex == []:
        lex = line.split()
    else:
        tree = line.split()
        if tree.count('RV') == 0:
            # Only retain sentences with 1 or 0 RVs
            filler_lex.append(' '.join(lex))
            filler_trees.append(' '.join(tree))
        elif tree.count('RV') == 1 and tree.count('WAS') == 1:
            optiona_lex.append(' '.join(lex))
            optiona_trees.append(' '.join(tree))
        elif tree.count('RV') == 1 and tree.count('WAS') == 0:
            optionb_lex.append(' '.join(lex))
            optionb_trees.append(' '.join(tree))
        lex = []
        tree = []

if len(optiona_lex) < len(optionb_lex):
    smaller_option = 'A'
    actual_smaller_len = len(optiona_lex)
    actual_larger_len = len(optionb_lex)
else:
    smaller_option = 'B'
    actual_smaller_len = len(optionb_lex)
    actual_larger_len = len(optiona_lex)

if actual_larger_len < proportion * actual_smaller_len:
    # need to trim the smaller list
    smaller_len = int(float(actual_larger_len) / proportion)
    larger_len = actual_larger_len
else:
    # need to trim the larger list
    larger_len = actual_smaller_len * proportion
    smaller_len = actual_smaller_len

smaller_indices = np.arange(actual_smaller_len)
larger_indices = np.arange(actual_larger_len)
random.shuffle(smaller_indices)
smaller_indices = smaller_indices[:smaller_len]
random.shuffle(larger_indices)
larger_indices = larger_indices[:larger_len]

if smaller_option == 'A':
    optiona_lex = list(np.array(optiona_lex)[smaller_indices])
    optiona_trees = list(np.array(optiona_trees)[smaller_indices])
    optionb_lex = list(np.array(optionb_lex)[larger_indices])
    optionb_trees = list(np.array(optionb_trees)[larger_indices])
else:
    optionb_lex = list(np.array(optionb_lex)[smaller_indices])
    optionb_trees = list(np.array(optionb_trees)[smaller_indices])
    optiona_lex = list(np.array(optiona_lex)[larger_indices])
    optiona_trees = list(np.array(optiona_trees)[larger_indices])

filler_lex.extend(optiona_lex)
filler_lex.extend(optionb_lex)
filler_trees.extend(optiona_trees)
filler_trees.extend(optionb_trees)
output_indices = np.arange(len(filler_lex))
random.shuffle(output_indices)

for sent_i in output_indices:
    print(filler_lex[sent_i])
    print(filler_trees[sent_i])

    
