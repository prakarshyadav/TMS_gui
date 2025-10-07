import DuoMAG
import time

#First TMS, TTL Out going into TTL In on the worker
controller = DuoMAG.DuoMAG("COM9")
worker = DuoMAG.DuoMAG("COM10")


controller.Pulse(65)

worker.Pulse(60)

controller.TTL(100, 5)

controller.Pulse()

#worker will pulse 500ms (100 * 5ms) after the controller

