import pandas as pd

docA = "The cat sat on my face"
docB = "The dog sat on my bed"

bowA = docA.split(" ")
bowB = docB.split(" ")

print(bowB)

wordSet = set(bowA).union(set(bowB))

print(wordSet)

wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)

print(wordDictA)

for word in bowA:
	wordDictA[word] += 1

for word in bowB:
	wordDictB[word] += 1

print(wordDictA)

import pandas as pd
pd.DataFrame([wordDictA, wordDictB])

def computeTF(wordDict, bow):
	tfDict = {}
	bowCount = len(bow)
	for word, count in wordDict.items():
		tfDict[word] = count / float(bowCount)
	return tfDict

tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)

print(tfBowA)
print(tfBowB)

def computeIDF(docList):
	import math
	idfDict = {}
	N = len(docList)
	
	idfDict = dict.fromkeys(docList[0].keys(), 0)
	for doc in docList:
		for word, val in doc.items():
			idfDict[word] += 1
	
	for word, val in idfDict.items():
		idfDict[word] = math.log10(N / float(val))

	return idfDict

idfs = computeIDF([wordDictA, wordDictB])

def computeTFIDF(tfBow, idfs):
	tfidf = {}
	for word, val in tfBow.items():
		tfidf[word] = val * idfs[word]
	return tfidf

tfidfBowA = computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)

pd.DataFrame([tfidfBowA, tfidfBowB])
