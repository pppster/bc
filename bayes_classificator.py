# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:25:40 2021

@author: Fabian
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class BayesClassificator:
    def Train(self, XTrain, YTrain):
        """
        Trainiert den naiven Bayes Klassifikator zum Klassifizieren von zwei Klassen mit variabler Merkmalszahl miitels Gaussscher Normalverteilung
        
        
        Parameters
        ----------
        XTrain : Pandas DataFrame
            Trainingsmerkmale, variable Anzahl von Merkmalen.
        YTrain : Pandas DatraFrame
            Dataframe aus Trainingszielvektor mit zwei Kategorien

        Returns
        -------
        None.

        """
        # Merkmalsmatrix und Zielvektor vereinigen für einfache Gruppierung nach Category
        self.df = pd.concat([XTrain, YTrain], axis = 1)
        
        # Anzahl der Features
        _, self.nFeatures = XTrain.shape
        # Anzahl der Klassen
        self.Cats= pd.unique(self.df["Category"])
        
        # Anzahl der Samples der jeweiligen Kategorien
        self.nCatSamples = self.df["Category"].value_counts()
        
        # Mittelwert und Varianz der Merkmale für jede Category
        self.mean = self.df.groupby("Category").mean()
        self.var = self.df.groupby("Category").var(ddof=0)
        # Priori Wahrscheinlichkeit der Klassenhäufigkeit
        self.priors = self.nCatSamples / float(self.df.shape[0])
        
    def Predict(self, testDataFrame):
        """
        Klassifiziert die übergebenen testDaten

        Parameters
        ----------
        testDataFrame : Pandas DataFrame
            Testmerkmale.

        Returns
        -------
        yPred : list of int
            Liste der vorhergesagte Klassenzugehörigkeit.

        """
        # Jede Reihe im testDataFrame durchgehen und Vorhersage treffen
        yPred = [self._Predict(x) for x in testDataFrame.values]
        return yPred
    
    def _Predict(self, x):
        """
        Hilfsfunktion zur Berechnung der Klassenzugehörigkeit

        Parameters
        ----------
        x : Merkmalsvektor

        Returns
        -------
        int
            vorhergesagte Klassenzugehörigkeit.

        """
        # posterior = (prior * likelihood) / evidence
        
        # Logarithmus der Prioriwahrscheinlichkeiten aus den Trainingsdaten
        # -> log(prior * likelihood) -> log(prior) + log(likelihood), vermeidet Underflow
        prior = np.log(self.priors)
        
        # Likelihood über alle Merkmale
        likelihood = np.log(np.exp(- (x-self.mean)**2 / (2*self.var)) / np.sqrt(2*np.pi * self.var))
        
        # Summieren der Likelihood
        likelihoodSum = likelihood.sum(axis = 1)
        posterior = prior + likelihoodSum
        # evidence ist unabhängig der Klasse -> konstant und kann deswegen vernachlässigt werden
        
        # vorhergesagte Klassenzugehörigkeit auf Basis der größten Posterior Wahrscheinlichkeit zurückgeben
        return posterior.values.argmax()
        
        

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Daten auslesen und Vorbereiten
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

dataFrame = pd.read_excel("Merkmale_ext.xlsx")

# Nicht benötigte Spalten entfernen
df = dataFrame.drop(columns = "PicNo")

# Kategorien wählen
df = pd.DataFrame(df, columns= ['CentroidRatio','AspectRatio', 'HullBBAreaRatio','ConturBBAreaRatio', 'ClosedContourCount','ContConv','Category'])

# Kategorien numerisieren
catsToInt = {"Brush": 0, "Comb": 1}
df["Category"] = df["Category"].map(catsToInt)

# In Merkmalsmatrix und Zielvektor aufteilen
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
        
# In Trainings und Testdaten splitten
randomSeed = 42
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state = randomSeed)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
naive Bayes Klassifikator instantiieren und und mit Trainingsdaten trainieren
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

naiveBayes = BayesClassificator()
naiveBayes.Train(XTrain, yTrain)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Testdaten eingeben und auswerten
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

yPred = naiveBayes.Predict(XTest)

print("------------------------ Gausscher naive Bayes Klassifikator ------------------------")
print("Anzahl Merkmale: " + str(XTrain.shape[1]))
print("Anzahl Traingsdaten: " + str(XTrain.shape[0]))
print("Anzahl Testdaten: " + str(XTest.shape[0]))
print("Kategorien: " + str(pd.unique(dataFrame["Category"])))
# Klassifikationsreport
classificationRep = classification_report(yTest, yPred, target_names=("Brush", "Comb"), output_dict=True)
print("------------ KlassifikationsclassificationRep ------------")
print(classification_report(yTest, yPred, target_names=("Brush", "Comb")))

#### Auswertung als Excel speichern
writer = pd.ExcelWriter("Klassifikationsreport_NaiveBayes.xlsx", engine='xlsxwriter')

# Klassifikationsreport als pd.Dataframe vorbereiten
classificationRepDf = pd.DataFrame(classificationRep)
classificationRepDf = classificationRepDf.transpose()

# KLassifikationsreport und Konfusionsmatrix
classificationRepDf.to_excel(writer, sheet_name="Report")


writer.save()