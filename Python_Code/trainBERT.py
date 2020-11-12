from textpipe import doc, pipeline

file = open('/Models/starGEO.txt', mode = 'r')

sample_text = file.read()
sample_text = " <!DOCTYPE>"

document = doc.Doc(sample_text)
print(document.clean)
print(document.language)
print(document.nwords)
pipe = pipeline.Pipeline(['CleanText', 'NWords'])
print(pipe(sample_text))
