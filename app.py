import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec
import numpy as np
import re
import streamlit as st

st.title("NLP")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

#lemmatization
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        tempList = []
        for word2 in word:
            tempList.append(wordlemmatizer.lemmatize(word2))
        lemmatized_words.append(tempList)
    return lemmatized_words

#Uniquewords
def uniqueWord(w):
    w2=[]
    for word in w:
        tempList=[]
        for word2 in word:
            if tempList.count(word2)<1:
                    tempList.append(word2)
        w2.append(tempList)
    return w2

#removing special characters

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text
#Removing Stopwords
def removeStopWord(word_text):
    filtered_sentence = [] 
    stop_words = set(stopwords.words('english'))   
    for w in word_text:
        tempList=[]
        for x in w:
            if x.lower() not in stop_words: 
                tempList.append(x)
        filtered_sentence.append(tempList)
    return filtered_sentence
def meanOfWord(model, sentence):
    posList=['CD']
    nounList=['NN','NNP','NNS','NNPS']
    value=[]
    count=0
    noun=0
    for word in sentence:
        a=model.wv.similar_by_word(word)
        temp=[]
        for w in a:
            temp.append(w[1])
        posValue=nltk.pos_tag([word])
        wordScore=np.mean(temp)
        if posValue[0][1] in posList:
            count=count+1
        else:
            valueIfNum=checkNum(word)
            count=count+valueIfNum
        if posValue[0][1] in nounList:
            noun=noun + .25
        value.append(wordScore)
    return np.mean(value)+count+noun
def checkNum(s):
    l= ['1','2','3','4','5','6','7','8','9','0']
    check =False

    for i in s:
        if i in l:
            check = True
            break
    if check == True:
        return 1
    else:
        return 0
    # lets begin ;) 

Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
text= st.text_input("Write your text\n")
sentences = sent_tokenize(text)
text_noSpecial_character = remove_special_characters(str(text))
word_text = [[text_noSpecial_character for text_noSpecial_character in sentences.split()] for sentences in sentences]
stop_text= removeStopWord(word_text)
unique_text= uniqueWord(stop_text)  
lemma_text = lemmatize_words(unique_text)

model = Word2Vec(lemma_text, min_count=1,sg=1)

score=[]
for index, sentence in enumerate(lemma_text):
    i = lemma_text.index(sentence)
    meanScore= meanOfWord(model,sentence)
    temp = [i,meanScore]
    score.append(temp)

st.write(score)
