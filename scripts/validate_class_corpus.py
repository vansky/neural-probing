'''
Ensures that a classifier file has equal numbers of 
classes and tokens for each line. If not,
The offending token and class lines are output.
'''
import sys

lex = []
for line in sys.stdin:
    if lex == []:
        lex = line.strip().split()
    else:
        classes = line.strip().split()
        if len(lex) != len(classes):
            print(' '.join(lex))
            print(' '.join(classes))
        lex = []
