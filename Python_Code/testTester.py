from tester import tester

t = tester()

t.whichQuery(1)

series = ["GSE47860",
        "GSE47861",
        "GSE72295",
        "GSE27175",
        "GSE19829",
        "GSE55699",
        "GSE60995",
        "GSE60494",
        "GSE73923",
        "GSE121382",
        "GSE69428",
        "GSE69429",
        "GSE92241",
        "GSE51260",
        "GSE133987",
        "GSE56443",
        "GSE66387",
        "GSE76360"]

score = t.returnScore(series)

print(score)
