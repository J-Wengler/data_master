import os
from tester import tester
import math
import sys
resultPath = ["MultipartitieRank", "PositionRank", "SingleRank", "TFIDF", "TextRank", "TopicalRank", "YAKE", "KPMINER", "TopicRankResults"]

bestCombo = open('/Models/bestCombo.txt', 'w+')


def getScore(filePath):
    score_to_series = {}
    with open(filePath, "r+") as in_file:
        for line in in_file:
            line_list = line.split('-')
            score_to_series[line_list[1].strip()] = line_list[0].strip()
    ##########################
    #Get top 10, 20, 50, 100, 200 results and get a percentage
    scores = []
    #toTest = [1,10,100,1000,10000]
    toTest = [10,20,50,100,200]
    #toTest = [1,10,800]
    #print()
    #print("SCORE_TO_SERIES LENGTH : {}".format(len(score_to_series)))
    #print()
    for topNum in toTest:
        topResults = getXTopResults(topNum, score_to_series)
        myTester = tester()
        filePathList = filePath.split('/')
        queryNum = int(filePathList[4][0])
        myTester.whichQuery(queryNum)
        score = myTester.returnPercent(topResults)
        print(score)
        scores.append(score)
    return scores


def getXTopResults(x, series_to_score):
    copy_of_keys = series_to_score.keys()
    sorted_keys = sorted(copy_of_keys, reverse = True)
    to_return = []
    print(len(sorted_keys))
    if len(sorted_keys) < 100:
        return [0,0,0,0,0]
    else:
        for per in range(0,x):
            #print("PERCENT : {}".format(per))
            #print("SORTED KEYS LENGTH: {}".format(len(sorted_keys)))
            to_return.append(series_to_score[sorted_keys[per]])

        return to_return

    

def findAvgPercent(series_to_score):
    avg_per = 0
    num_keys = 0
    for key in series_to_score:
        if series_to_score[key] != '':
            temp_per = float(series_to_score[key])
            if(math.isnan(temp_per) is False):
                avg_per += float(series_to_score[key])
            num_keys += 1
    avg_to_return = avg_per / num_keys
    return avg_to_return


#outputFile = open('/Models/BioWordVecOutput.txt', 'w+')
#outputFile.write("BioWordModel Results\n")
#outputFile.write("MODEL\tQUERY\tSCORE\n")
#for name in resultPath:
#    path = "/Models/BIOWORDVEC/{}".format(name)
#    for fileName in os.listdir(path):
#        score = getScore(path + "/" + fileName)
#        query = fileName[0]
#        strForFile = "{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(), str(score).strip())
#        outputFile.write(strForFile)
#
#outputFile = open('/Models/FastTextWikiOutput.txt', 'w+')
#outputFile.write("FASTTEXTWIKI Results\n")
#outputFile.write("MODEL\tQUERY\tSCORE\n")
#for name in resultPath:
#    path = "/Models/FastTextWiki/{}".format(name)
#    for fileName in os.listdir(path):
#        score = getScore(path + "/" + fileName)
#        query = fileName[0]
#        strForFile = "{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(), str(score).strip())
#        outputFile.write(strForFile)
#outputFile = open('/Models/FastTextCBOWOutput.txt', 'w+')
#outputFile.write("FASTTEXTCBOW Results\n")
#outputFile.write("MODEL\tQUERY\tSCORE\n")
#for name in resultPath:
#    path = "/Models/FastTextWiki/{}".format(name)
#    for fileName in os.listdir(path):
#        score = getScore(path + "/" + fileName)
#        query = fileName[0]
#        strForFile = "{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(), str(score).strip())
#        outputFile.write(strForFile)


outputFile = open('/Models/FastTextSKIPGRAMOutput.txt', 'w+')
outputFile.write("FASTTEXTSKIPGRAM Results\n")
outputFile.write("MODEL\tQUERY\t#\tSCORE\n")

for name in resultPath:
    path = "/Models/FastTextSkipGram/{}".format(name)
    top_nums = [10, 20, 50, 100, 200]
    for fileName in os.listdir(path):
        scores = getScore(path + "/" + fileName)
        for i,score in enumerate(scores): 
            query = fileName[0]
            if(top_nums[i] == 100):
                bestCombo.write("FTSKIPGRAM\t{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip()))
            strForFile = "{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip())
            outputFile.write(strForFile)

print("{}% percent complete |{}{}| \r".format(1/6*100, '-','     '))
outputFile = open('/Models/FastTextCBOWOutput.txt', 'w+')
outputFile.write("FASTTEXTCBOW Results\n")
outputFile.write("MODEL\tQUERY\t#\tSCORE\n")

for name in resultPath:
    path = "/Models/FastTextCBOW/{}".format(name)
    top_nums = [10, 20, 50, 100, 200]
    for fileName in os.listdir(path):
        scores = getScore(path + "/" + fileName)
        for i,score in enumerate(scores): 
            query = fileName[0]
            if(top_nums[i] == 100):
                bestCombo.write("FTCBOW\t{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip()))
            strForFile = "{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip())
            outputFile.write(strForFile)
sys.stdout.flush()
print("{}% percent complete |{}{}| \r".format(2/6*100, '--','    '))


outputFile = open('/Models/FastTextWikiOutput.txt', 'w+')
outputFile.write("FASTTEXTWIKI Results\n")
outputFile.write("MODEL\tQUERY\t#\tSCORE\n")

for name in resultPath:
    path = "/Models/FastTextWiki/{}".format(name)
    top_nums = [10, 20, 50, 100, 200]
    for fileName in os.listdir(path):
        scores = getScore(path + "/" + fileName)
        for i,score in enumerate(scores):
            if(top_nums[i] == 100):
                bestCombo.write("FTWIKI\t{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip()) )         
            query = fileName[0]
            strForFile = "{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip())
            outputFile.write(strForFile)
sys.stdout.flush()
print("{}% percent complete |{}{}| \r".format(3/6*100, '---','   '))


outputFile = open('/Models/BioWordVecOutput.txt', 'w+')
outputFile.write("BIOWORDVEC Results\n")
outputFile.write("MODEL\tQUERY\t#\tSCORE\n")

for name in resultPath:
    path = "/Models/BIOWORDVEC/{}".format(name)
    top_nums = [10, 20, 50, 100, 200]
    for fileName in os.listdir(path):
        scores = getScore(path + "/" + fileName)
        for i,score in enumerate(scores): 
            query = fileName[0]
            if(top_nums[i] == 100):
                bestCombo.write("BIOWORDVEC\t{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip()))
            strForFile = "{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip())
            outputFile.write(strForFile)
sys.stdout.flush()
print("{}% percent complete |{}{}| \r".format(4/6*100, '----','  '))
outputFile = open('/Models/SpacyOutput.txt', 'w+')
outputFile.write("SPACY Results\n")
outputFile.write("MODEL\tQUERY\t#\tSCORE\n")

for name in resultPath:
    path = "/Models/SpacyWebLG/{}".format(name)
    top_nums = [10, 20, 50, 100, 200]
    for fileName in os.listdir(path):
        scores = getScore(path + "/" + fileName)
        for i,score in enumerate(scores):
            query = fileName[0]
            if(top_nums[i] == 100):
                bestCombo.write("SPACY\t{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip()))
            strForFile = "{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip())
            outputFile.write(strForFile)
sys.stdout.flush()
print("{}% percent complete |{}{}| \r".format(5/6 * 100, '-----',' '))
outputFile = open('/Models/SciSpacyOutput.txt', 'w+')
outputFile.write("SCISPACY Results\n")
outputFile.write("MODEL\tQUERY\t#\tSCORE\n")

for name in resultPath:
    path = "/Models/SciSpacy/{}".format(name)
    top_nums = [10, 20, 50, 100, 200]
    for fileName in os.listdir(path):
        scores = getScore(path + "/" + fileName)
        for i,score in enumerate(scores): 
            query = fileName[0]
            if(top_nums[i] == 100):
                bestCombo.write("SCISPACY\t{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip()))
            strForFile = "{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(),str(top_nums[i]).strip(), str(score).strip())
            outputFile.write(strForFile)

sys.stdout.flush()
print("{}% percent complete |{}{}| \r".format(6/6*100, '------',''))
