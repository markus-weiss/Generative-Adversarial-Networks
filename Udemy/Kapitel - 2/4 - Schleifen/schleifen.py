### Schleifen in Python ###

# 0 - 5
#
for i in range(5):
    print(i , 'Penis')

# Mit Start Stop und Step
for i in range(1 , 10 , 2):
    print(i , 'Penis')

zahl = 10 
a = False

while a == False:
    print(zahl)
    zahl -= 1
    if zahl <= 0:
        a = True
        print('NoMoreMoney')
