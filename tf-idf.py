from textblob import TextBlob
from nltk.corpus import stopwords
import pandas as pd
import math

url = 'https://docs.google.com/spreadsheets/d/1e-CfY1uUYLpU8GMmWWW77hT1GbpLssfjxWIUJz3e_Ag/export?gid=1814555242&format=csv'

df = pd.read_csv(url)

positive_comments = df[df.columns[-3]].tolist()

blob_list = [TextBlob(str(comment)) for comment in positive_comments]


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)


def idf(word, bloblist):
    return math.log(
        len(bloblist) / (
            1 + n_containing(word, bloblist)
        )
    )


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


words = {}

stopwords = stopwords.words('english')

for blob in blob_list:
    for word in blob.words:
        lower_word = word.lower()
        if lower_word in words:
            words[lower_word] += 1
        elif lower_word not in stopwords:
            words[lower_word] = 1
        else:
            continue

del words['nan']
del words['none']
