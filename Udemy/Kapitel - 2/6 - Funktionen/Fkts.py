def bester_schüler(noten_klasse , konsolenausgabe= False):
    bis_jetzt_bester_schüler = ""
    bis_jetzt_bester_note = 0
    for name, note in noten_klasse.items():
        if note > bis_jetzt_bester_note:
            bis_jetzt_bester_note = note 
            bis_jetzt_bester_schüler = name
    if konsolenausgabe == True:
        print("Ausgabe: ", bis_jetzt_bester_schüler,bis_jetzt_bester_note)
    return bis_jetzt_bester_schüler, bis_jetzt_bester_note



noten_klasse_8a = {"armin" : 1, "ben": 2, "jan": 1,"armin" : 3, "ben": 4, "jan": 2}

name, note = bester_schüler(
    noten_klasse_8a,
    True

)

# print(name,note)