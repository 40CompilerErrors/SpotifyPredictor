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


#From this class we should get the list of title words, lemmatized, for input, as well as the number of songs,
#listed by an identifier


class InputExtractor():
    def __init__(self, path):
        self.path = path

    def Extract_Input(self):
        songs = []
        words = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".json"):
                    json_file = (os.path.join(root, file))
                    print(json_file)
                    for s in self.Get_Songs(json_file):
                        songs.append(s)
                    for w in self.Tokenize(self.Get_Playlists(json_file)):
                        words.append(w)
        return set(songs), set(words)

    def Get_Songs(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            file_readed = file.read()
            jsonFile = json.loads(file_readed)
            jsonURL = []
            for playlist in jsonFile['playlists']:
                for song in playlist['tracks']:
                    jsonURL.append(song['track_uri'])
            return list(OrderedDict.fromkeys(jsonURL))

    def Get_Playlists(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            file_readed = file.read()
            jsonFile = json.loads(file_readed)
            playlists = []
            for pl in jsonFile['playlists']:
                playlists.append(pl['name'])
            return list(OrderedDict.fromkeys(playlists))

    def Tokenize(self,playlist):
        lematizer = WordNetLemmatizer()
        stopWords = set(stopwords.words('english'))

        tokenizado = []
        for title in playlist:
            tokens = word_tokenize(title)
            for word in tokens:
                if word not in stopWords:
                    tokenizado.append(word)

        lematizado = []
        for token in list(OrderedDict.fromkeys(tokenizado)):
            lematizado.append(lematizer.lemmatize(token))

        return list(OrderedDict.fromkeys(lematizado))

    def getDataframe(self):
        df = pd.DataFrame()
        df['Songs'] = self.data['Songs']
        df.drop_duplicates('first', True)
        return df

    def exportPlayList(self):
        with open('Songs.txt', 'w', encoding='utf-8') as file:
            for song in self.data['Songs']:
                file.write(song)
                file.write('\n')
