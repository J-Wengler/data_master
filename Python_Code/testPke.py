import pke
import os

extractors = [1,2,3,4,5,6,7,8,9]

#for ex in extractors:
path = '/Models/Queries/q1/'
for filename in os.listdir(path):
    if filename != 'names.txt':
        #print("FILENAME : {}".format(filename))
        #in_string = '/Models/Queries/q5/{}'.format(filename)
        #in_file = open(in_string, 'r')
        #contents = in_file.read()
        #contents = contents.replace('\n', ' ')
        #print("INPUT STRING : {}".format(contents))
        for ex in extractors:
            method = pke.unsupervised.TopicRank()
            if ex == 1:
                method = pke.unsupervised.TopicRank()
            elif ex == 2:
                method = pke.unsupervised.TfIdf()
            elif ex == 3:
                method = pke.unsupervised.KPMiner()
            elif ex == 4:
                method = pke.unsupervised.YAKE()
            elif ex == 5:
                method = pke.unsupervised.TextRank()
            elif ex == 6:
                method = pke.unsupervised.SingleRank()
            elif ex == 7:
                method = pke.unsupervised.TopicalPageRank()
            elif ex == 8:
                method = pke.unsupervised.PositionRank()
            elif ex == 9:
                method = pke.unsupervised.MultipartiteRank()
            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
            method.candidate_selection()
            method.candidate_weighting()
            keyphrases = method.get_n_best(n=10)
            print()
            print("METHOD : {}".format(method))
            print(keyphrases)
            print()


