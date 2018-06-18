import json
import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import OrderedDict
import csv
import pandas as pd

class ModelGenerator():
    def __init__(self, songs, words, path):
        self.songs = songs
        self.words = words
        self.path = path

    def CreateFrame(self):
        col = ["Words","Training","Songs"]
        rows = self.GetRows()
        df = pd.DataFrame(columns = col, data = rows)
        return df


    def GetRows(self):
        rows = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".json"):
                    json_file = (os.path.join(root, file))
                    print(json_file)
                    with open(json_file, 'r', encoding='utf-8') as file:
                        file_readed = file.read()
                        jsonFile = json.loads(file_readed)
                        for playlist in jsonFile['playlists']:
                            rows.append(self.FillRow(playlist))
        return rows

    def GetColums(self):
        columns = []
        for w in self.words:
            columns.append(w)
        for s in self.songs:
            columns.append(s)
        return columns

    def FillRow(self,playlist):
        row = []
        words = []
        training = []
        songs = []

        title = self.Lematize(playlist['name'])

        for w in self.words:
            if w in title:
                words.append(1)
                #print("Match found")
            else:
                words.append(0)

        for url in self.songs:
            found = 0
            for t in playlist['tracks']:
                if url in t['track_uri']:
                    found = 1
            if found:
                songs.append(1)
                #print("Match found")
            else:
                songs.append(0)

        #Crear un set de training
        training = self.createTraining(songs = songs)
        row.append(words)
        row.append(training)
        row.append(songs)

        return row

    def createTraining(self, songs):
        training = []


        return training

    def Lematize(self,title):
        lematizer = WordNetLemmatizer()
        stopWords = set(stopwords.words('english'))
        tokens = (w for w in word_tokenize(title) if w not in stopWords)
        title_lematized = []
        for token in list(OrderedDict.fromkeys(tokens)):
            title_lematized.append(lematizer.lemmatize(token))
        return set(title_lematized)