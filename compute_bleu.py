import nltk
import codecs
import os

infile = open('100best_nounk.out','r').readlines()
refs=[]
hyps=[]
for i in xrange(len(infile)/101):
    ref = [infile[i*101].strip().split()]
    refs.append(ref)
    best_sent = infile[i*101+1].strip().split()
    best_bleu = nltk.translate.bleu_score.sentence_bleu(ref,best_sent)
    for j in xrange(1,100):
        hyp = infile[i*101+j+1].strip().split()
        bleu = nltk.translate.bleu_score.sentence_bleu(ref,hyp)
        if (bleu > best_bleu):
            best_bleu = bleu
            best_sent = hyp
    hyps.append(best_sent)
print nltk.translate.bleu_score.corpus_bleu(refs,hyps)
