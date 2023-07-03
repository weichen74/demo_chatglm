'''
import nltk.data
print(nltk.data.path)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
'''
import os
import shutil
import random
import time
dict={"query1":"one","query2":"two","query3":"three"}

for i in dict:
    print(i,dict[i])

#dict.items()
for k,v in dict.items():
    print(k,v)

for k,v in zip(dict.keys(),dict.values()):
    print(k,v)

for i in dict:
    print(i)
print(dict["query1"])
