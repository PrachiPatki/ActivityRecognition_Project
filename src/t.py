import threading
import time
def temp(t):
    time.sleep(1)
    print(t)
    time.sleep(1)

tis =[]
for i in range(0,5):    
    t = threading.Thread(target = temp, args= (i,))
    tis.append(t)
    t.start()

for t in tis:
    t.join();

print('is here')