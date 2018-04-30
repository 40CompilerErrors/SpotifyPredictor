import InputExtractor as IE
import ModelGenerator as MG
import pandas as pd

if __name__ == '__main__':

    #Paso 1: Conseguir las columnas (words + songs)
    words = []
    songs = []

    path = r'H:\Workshop\Machine Learning and AI\Datasets\spotify\jsons'
    data = IE.InputExtractor(path)
    songs, words = data.Extract_Input()
    print("Total songs: " + str(len(songs)))
    print("Total words: " + str(len(words)))

    #Paso 2: Crear el dataframe de Pandas

    model = MG.ModelGenerator(songs, words, path)
    df = model.CreateFrame()


    #Paso 3: Crear la Neural Network

