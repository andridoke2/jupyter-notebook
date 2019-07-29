from sklearn.feature_extraction.text import TfidfVectorizer
import operator
import pandas as pd

corpus = [
  "this is my car which is excellent",
  "good car gives good milage",
  "this car is very expensive",
  "the boy is very good but can not drive a car",
  "this company is very rich"
]

df_train = pd.read_csv('data/sentiment-analysis-dataset.csv')
# print(df_train)

# take sample feture
sample_df_train = df_train[['Sentiment','SentimentText', 'ItemID', 'SentimentSource']]
# print(sample_df_train.head())

# clean data
sample_df_train = sample_df_train.dropna()

y_sample = sample_df_train['ItemID']
sample_df_train = sample_df_train.drop('ItemID', axis = 1)
# print(sample_df_train)

y_sample = sample_df_train['SentimentSource']
sample_df_train = sample_df_train.drop('SentimentSource', axis = 1)
# print(sample_df_train)

sentimentText = {}
for word in sample_df_train:
  sentimentText = sample_df_train['SentimentText']
  sentiment = sample_df_train['Sentiment']

# print(sentimentText)
# print(sentiment)

vocabulary = set()
for doc in sentimentText:
  vocabulary.update(doc.split())

print(vocabulary)

vocabulary = list(vocabulary)
word_index = {w: id for id, w in enumerate(vocabulary)}

print(word_index)

tfidf = TfidfVectorizer(vocabulary = vocabulary)
tfidf.fit(sentimentText)
tfidf.transform(sentimentText)
for doc in sentimentText:
  score = {}
  print(doc)
  print
  X = tfidf.transform([doc])
  for word in doc.split():
    score[word] = X[0, tfidf.vocabulary_[word]]
  sortedscore = sorted(score.items(), key = operator.itemgetter(1), reverse = True)
  print(sortedscore)
  print