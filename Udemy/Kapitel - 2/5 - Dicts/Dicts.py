### Dicts in Python ###


# key: value 

noten_klasse_8a = {"armin" : 1, "ben": 2, "jan": 1}

armins_note = noten_klasse_8a["armin"]
print("Armins Note: " , armins_note)

print("\n")

# index , inhalt 
# string, inhalt 
# key, value
#name, note

for schüler, note in noten_klasse_8a.items():
    print(schüler,' - ' ,note)

print("\n")

# Einzelne Abfage
for schüler in noten_klasse_8a.keys():
    print(schüler,' - ' ,'noten_klasse_8a[schüler]')

print("\n")

# Einzelne Abfage
for note in noten_klasse_8a.values():
    print(note)