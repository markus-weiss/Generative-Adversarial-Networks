### Moduel in Python 

# Import Funktion
from students import bester_schüler as bs
# Import all Funktions
from students import *
# Import all Funktions
# from students as st



noten_klasse_8a = {"armin" : 1, "ben": 2, "jan": 1,"armin" : 3, "ben": 4, "jan": 2}

name, note = bs(
    noten_klasse_8a,
    konsolenausgabe=True,
)

import random

zahl = random.randint(1,10)
print(zahl)