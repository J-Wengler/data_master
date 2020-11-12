library(tidyverse)

path = "/Users/jameswengler/fauxData.tsv"

data = read_delim(path, delim = "     ", skip = 2, col_names = c('Model', 'Query', 'Score'))

models = pull(data, Model)
queries = as.numeric(pull(data, Query))
scores = as.numeric(pull(data, Score))

clean_data = as_tibble(cbind(models, queries, scores))

#print(clean_data)

query1 = filter(clean_data, queries == "1")
query2 = filter(clean_data, queries == "2")
query3 = filter(clean_data, queries == "3")
query4 = filter(clean_data, queries == "4")
query5 = filter(clean_data, queries == "5")

query1 = mutate(query1, scores = as.numeric(scores))
query2 = mutate(query2, scores = as.numeric(scores))
query3 = mutate(query3, scores = as.numeric(scores))
query4 = mutate(query4, scores = as.numeric(scores))
query5 = mutate(query5, scores = as.numeric(scores))

print(clean_data)

ggplot(query1, aes(x=models, y=scores)) +
  theme(axis.text.x = element_text(angle = 90)) +
  ylim(0,1) +
  geom_point()
ggsave("Query1.png")

ggplot(query2, aes(x=models, y=scores)) +
  theme(axis.text.x = element_text(angle = 90)) +
  ylim(0,1) +
  geom_point()
ggsave("Query2.png")

ggplot(query3, aes(x=models, y=scores)) +
  theme(axis.text.x = element_text(angle = 90)) +
  ylim(0,1) +
  geom_point()
ggsave("Query3.png")

ggplot(query4, aes(x=models, y=scores)) +
  theme(axis.text.x = element_text(angle = 90)) +
  ylim(0,1) +
  geom_point()
ggsave("Query4.png")

ggplot(query5, aes(x=models, y=scores)) +
  theme(axis.text.x = element_text(angle = 90)) +
  ylim(0,1) +
  geom_point()
ggsave("Query5.png")
