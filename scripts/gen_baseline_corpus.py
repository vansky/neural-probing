import sys
import random

possible_tokens = ['ABC','DEF','HGI','JKL','MNO','PQR','STU','VWX','YZ']

lex = []

for line in sys.stdin:
    if lex == []:
        lex = line.strip()
    else:
        print(lex)
        FIRST = True
        for word in lex.split():
            if FIRST:
                sys.stdout.write(random.choice(possible_tokens))
                FIRST = False
            else:
                sys.stdout.write(' '+random.choice(possible_tokens))
        sys.stdout.write('\n')
        lex = []
