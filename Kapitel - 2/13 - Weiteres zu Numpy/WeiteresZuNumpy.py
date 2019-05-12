import numpy as np
import matplotlib.pyplot as plt

spieler = [1,2,3,4,5,6,7,8,9]

# Zufallszahlen von [1 - 10], Anzahl
zahlen = np.random.randint(1, 11, 5)
print(zahlen)

# Zufalls-Elemente Array und anzahl der auswahl
gewinner = np.random.choice(spieler, 5)

# Zufalls-reihenfolge Neuverteilung
spieler_ = np.random.permutation(spieler)
print(spieler_)

# Zufalls-zahlen (Normal Float)
zahlen = np.random.normal(size=10)
#zahlen = np.random.randn(10)
print(zahlen)
