### Matplotlib in Python ###
import matplotlib.pyplot as plt

noten_jan = [56,64,78,100,110]
noten_ben = [66,74,80,90,110]


# Linie Zeichnen
# x , y, color / range muss so lange sein wie das array #
plt.plot(range(10,15), noten_jan, color="blue")
plt.plot(range(10,15), noten_ben, color="red")
plt.legend(["Jan","Ben"])
plt.xlabel("x Achse")
plt.ylabel("y Achse")
plt.title("Grafik")
plt.show()


# Punkte anzeigen
# # x , y , color 
# x = [4,2,10,7]
# y = [10,4,9,3]
# plt.scatter(x,y, color="blue")
# plt.show()
