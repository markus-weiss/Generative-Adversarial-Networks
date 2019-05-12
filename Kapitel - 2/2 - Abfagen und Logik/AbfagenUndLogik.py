binIchPleite = None
binIchReich = None
kontostand = 0.0

if kontostand <= 0:
    binIchPleite = False
    print('Dumm gelaufen')
    if kontostand > 1000:
        binIchReich = True
    else:
        binIchReich: False
else: binIchPleite= True