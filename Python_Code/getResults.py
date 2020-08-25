emport os
from tester import tester

resultPath = ["MultipartitieRank", "PositionRank", "SingleRank", "TFIDF", "TextRank", "TopicalRank", "YAKE", "KPMINER", "TopicRankResults"]

def getScore(filePath):
    series_to_score = {}
    with open(filePath, "r+") as in_file:
        for line in in_file:
            line_list = line.split('-')
            series_to_score[line_list[0]] = line_ist[1]
    #avg_per = findAvgPercent(series_to_score)
    series_to_test = []
    for key in series_to_score:
        if series_to_score[key] > 95:
            series_to_test.append(key)
    myTester = tester()
    filePathList = filePath/split('/')
    queryNum = int(filePathList[3][0])
    myTester.whichQuery(queryNum)
    score = myTester.returnScore(series_to_test)
    return score



    

def findAvgPercent(series_to_score):
    avg_per = 0
    num_keys = 0
    for key in series_to_score:
        avg_per += series_to_score[key]
        num_keys += 1
    avg_to_return = avg_per / num_keys
    return avg_to_return


outputFile = open('/Models/BioWordVecOutput.txt', 'w+')
outputFile.write("BioWordModel Results\n\n")
outputFile.write("MODEL\t\t\QUERY\t\tSCORE\n")

for name in resultPath:
    path = "/Models/{}".format(name)
    for fileName in os.listdir(path):
        score = getScore(path + "/" + fileName)
        strForFile = "{}\t\t{}\t\t{}\n".format(name, fileName, score)
        outputFile.write(strForFile)



