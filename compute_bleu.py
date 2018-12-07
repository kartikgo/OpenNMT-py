import nltk
import codecs
import os

infile = open('10best_nounk.out.train','r').readlines()
refs=[]
hyps=[]
for i in xrange(len(infile)/11):
    ref = [infile[i*11].strip().split()]
    #refs.append(ref)
    kbest = []
    #print nltk.translate.bleu_score.sentence_bleu(ref,ref)
    print 1.0
    #best_sent = infile[i*11+1].strip().split()
    #best_bleu = nltk.translate.bleu_score.sentence_bleu(ref,best_sent)
    for j in xrange(10):
        hyp = infile[i*11+j+1].strip().split()
        bleu = nltk.translate.bleu_score.sentence_bleu(ref,hyp)
        print bleu
        #if (bleu > best_bleu):
        #    best_bleu = bleu
        #    best_sent = hyp
    #hyps.append(best_sent)
#print nltk.translate.bleu_score.corpus_bleu(refs,hyps)
