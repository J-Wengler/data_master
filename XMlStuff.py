from nltk.tokenize import sent_tokenize, word_tokenize

input_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/pubmed20n0001.xml")
input_text = input_file.read()


words = word_tokenize(input_text)

for word in words:
    if(word.startswith('<')):
        print(word)

