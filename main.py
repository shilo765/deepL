
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    games = pd.read_csv("Data//game.csv")
    pd.set_option('display.max_columns', None)
    print("data info:")
    print("")
    games.info()
    print("")
    print("")
    print("")
    #show the game top 5 lines
    print("the game top 5 lines:")
    print("")
    print(games.head())
    pd.reset_option("max_columns")

    # drop the lines  file name,lines. not help us to the goal
    games = games.drop(columns=['file_name', 'lines'])
    print("")
    print("")
    print("")
    # i remove the players with problematic values in the database/ null or zero values
    print("i remove the players with problematic values in the database/ null or zero values:")
    print("")
    games = games[games.player != 'Alekhine']
    games = games[games.player != 'Capablanca']
    games = games[games.player != 'Tal']
    games = games[games.player != 'Morphy']
    print(games)
    print("")
    print("")
    print("")
    #remove all game with elo=0. not currect elo cant be zero by ruls
    print("remove all elo==0")
    #sort the data by date and player
    games = games.sort_values(by=['date', 'player'])
    #fill all the null in 0
    games=games.fillna(0)
    games = games[games.opponent_Elo != 0]
    print(games)
    print("")
    print("")
    print("")
    # some of the player elo are missing so i go up all the data and change to the most privious game that has elo
    print("player elo fix:")
    #            [Botvinnik Carlsen Nakamura Caruana Polgar Kasparov Fischer Anand
    tempPlayer = [0        ,0      ,0       ,0      ,0     ,0       ,0      ,0     ]


    def f(x):
        return {
            'Botvinnik': 0,
            'Carlsen': 1,
            'Nakamura': 2,
            'Caruana': 3,
            'Polgar': 4,
            'Kasparov': 5,
            'Fischer': 6,
            'Anand': 7
        }.get(x, 9)
    for i in range(games.shape[0]):
        if (games.iat[i, 3] != 0):
            tempPlayer[f(games.iat[i, 0])] = games.iat[i, 3]
        if (games.iat[i, 3] == 0):
            games.iloc[i, 3] =  tempPlayer[f(games.iat[i, 0])]
    print("")
    print(games)

    # make color dummies
    color = pd.get_dummies(games.color)
    games = games.join(color)
    games = games.drop(columns=['color'])
    print("")
    print("")
    print("make color to dummies")
    print(games)

    #make the result as dummies 0 to lose 1 to draw and 2 to win
    result = pd.get_dummies(games.result)
    for i in range(games.shape[0]):
        if (result.iat[i, 2] == 1):
            games.iat[i, 4] = int(2)
        if (result.iat[i, 1] == 1):
            games.iat[i, 4] = int(0)
        if (result.iat[i, 0] == 1):
            games.iat[i, 4] = int(1)
    print("")
    print("")
    print("2 to win 1 to draw and 0 to lose")

    print(games['result'])

    #insert age
    games.insert(0, 'age', 0)
    for i in range(games.shape[0]):
        if (games.iloc[i, 1] == 'Anand'):
            games.iloc[i, 0] = int(games.iloc[i, 8][:4]) - 1969
        if (games.iloc[i, 1] == 'Botvinnik'):
            games.iloc[i, 0] = int(games.iloc[i, 8][:4]) - 1911
        if (games.iloc[i, 1] == 'Carlsen'):
            games.iloc[i, 0] = int(games.iloc[i, 8][:4]) - 1990
        if (games.iloc[i, 1] == 'caruana'):
            games.iloc[i, 1] = int(games.iloc[i, 8][:4]) - 1992
        if (games.iloc[i, 1] == 'Fischer'):
            games.iloc[i, 0] = int(games.iloc[i, 8][:4]) - 1943
        if (games.iloc[i, 1] == 'Kasparov'):
            games.iloc[i, 0] = int(games.iloc[i, 8][:4]) - 1963
        if (games.iloc[i, 1] == 'Nakamura'):
            games.iloc[i, 0] = int(games.iloc[i, 8][:4]) - 1987
        if (games.iloc[i, 1] == 'Polgar'):
            games.iloc[i, 0] = int(games.iloc[i, 8][:4]) - 1976
    print("")
    print("")
    print("the age of players:")
    print(games['age'])

    print("")
    print("")
    print("fix the age and drop the date")
    games = games[games.age > 0]
    games = games.drop(columns=['date'])
    print(games)
    #make the player to dummies
    print("")
    print("")
    print("players as dummies")
    player = pd.get_dummies(games.player)
    games = games.join(player)
    games = games.drop(columns=['player'])

    games = games.drop(columns=['site', 'event', 'opponent'])
    pd.set_option('display.max_columns', None)
    print(games.head())

    games["result"] = pd.to_numeric(games["result"])
    print("games info")
    print(games.info())

    #use deep learning
    y = games['result']
    X = games.drop('result', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)
    print("")
    print("ecisionTreeClassifier:")
    print(predictions)
    print("")
    print("")

    print(classification_report(y_test, predictions))
    confusion_matrix(y_test, predictions)

    model0 = 0
    modelD0=0
    modelW0=0
    modelL0=0
    countW=0
    countD=0
    countL=0
    for i in range(games.shape[0]):
        if (games.iloc[i, 1] - games.iloc[i, 2] >= 10 and (games.iloc[i, 3] == 2)):
            model0 = model0 + 1
            modelW0=modelW0+1
        if(games.iloc[i, 3] == 2):
            countW=1+countW
        if (games.iloc[i, 1] - games.iloc[i, 2] <= -10 and (games.iloc[i, 3] == 0)):
            model0 = model0 + 1
            modelL0=modelL0+1
        if(games.iloc[i, 3] == 0):
            countL=1+countL
        if (games.iloc[i, 1] - games.iloc[i, 2] < 10 and (games.iloc[i, 1] - games.iloc[i, 2] > -10) and (
                games.iloc[i, 3] == 1)):
            model0 = model0 + 1
            modelD0=modelD0+1
        if(games.iloc[i, 3]==1):
            countD=countD+1
    print("model 0:")
    print(model0/games.shape[0])
    print("wins:")
    print(modelW0/countW)
    print("loses :")
    print(modelL0/countL)
    print("draw :")
    print(modelD0 / countD)
    print('knn:')
    y = games['result']
    X = games.drop('result', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(classification_report(y_test, predictions))
    print('')
    print("randomForest:")
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    print(classification_report(y_test, predictions))
    print("svm:")
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    print(classification_report(y_test, predictions))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
