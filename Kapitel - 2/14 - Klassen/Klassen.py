import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, start_kontostand):
        self.kontostand = start_kontostand
        if self.kontostand > 0.0:
            self.bin_ich_pleite = False
        else:
            self.bin_ich_pleite = True

    def play(self):
        while self.bin_ich_pleite == False:
            print("Ich bin nicht pleite!" , self.kontostand)
            self.kontostand -= 1
            if self.kontostand <= 0:
                self.bin_ich_pleite = True

start_kontostand = 7
g = Game(start_kontostand)
g.play()
