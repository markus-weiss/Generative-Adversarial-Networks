### Schleifen für Listen in Python


# noten = [1 , 2, 3 ,1 ,2 , 3, 45, 6]

# for i in noten:
#     print(i)

# print("\n")


# # length of array = len(noten)
# for i in range(len(noten)):
#     print(noten[i])


# print("\n")

noten = [1 , 2, 3 ,1 ,2 , 3, 45, 6]
fächer  =["Mathe","Deutsch","English","Info","Geschichte","Kunst"]

for note, fach in zip(noten, fächer):
    print(note, " - " , fach)


präferenzen = ["Mathe","Deutsch","English","Info","Geschichte","Kunst"]

# for stelle, inhalt
for index, fach in enumerate(präferenzen):
    print("Das Fach: ", fach, " ist an Stelle ", index + 1)

