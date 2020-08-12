class tester:

    def __init__(self):
        self.query_number = 0
        self.data_names = [1,2,3,4]
        self.in_file = ""

    def whichQuery(self, query_num):
        self.query_number = query_num
        self.in_file = "/Models/Queries/q{}/names.txt".format(query_num)
        with open(self.in_file, "r") as names:
            for name in names:
                self.data_names.append(name.rstrip())


    def returnScore(self, data):
        #Compare ranks, model will rank all starGEO articles and calculate the average rank of the originial articles
        #information retrival metrics for accuracy
        #Precision and Recall
        num_obtained = len(data)
        num_relevant  = 0
        total_relevant = len(self.data_names)
        for name in data:
            if name in self.data_names:
                num_relevant += 1
        
        recall = num_relevant / total_relevant
        precision = num_relevant / num_obtained
        score = 2 * (precision * recall) / (precision + recall)
        return score


