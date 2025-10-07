import DuoMAG
import time

#First TMS, TTL Out going into TTL In on the worker
controller = DuoMAG.DuoMAG("COM9")
worker = DuoMAG.DuoMAG("COM10")


# controller.Pulse(50)

# worker.Pulse(60)

# controller.TTL(100, 5)

# controller.Pulse()

t = 1
while t >0:
    print(t)
    t0 = time.time()
    # controller.Pulse()
    controller.Pulse(5,True)
    print('stim 1', time.time()-t0)
    time.sleep(1)
    worker.Pulse(15,True)
    print('stim 2', time.time()-t0)
    time.sleep(2)

#worker will pulse 500ms (100 * 5ms) after the controller

