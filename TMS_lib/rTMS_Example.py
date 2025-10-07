import time

import DuoMAG

controller = DuoMAG.DuoMAG("COM9")

controller.Pulse(50)

numOfPulses = 10

for i in range(1, numOfPulses):
    controller.Pulse()
    time.sleep(2) # Wait 2 seconds between pulses