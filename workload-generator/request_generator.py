import json
from threading import Thread
import subprocess
from time import sleep
import numpy as np
import math

def func(a=1):
    command = f"hey -n {a} -c {a} -o 'csv' '${FUNCTION_URL}"  # Replace with your target URL
    subprocess.run(command, shell=True) 
    # pass

# read the unstable invocation data and normalise for our capacity
with open('./invokation_data.json')as f:
    data = json.load(f)

d = [int(i) for i in data]
# setting the random seed for replication of results
seed = 29
np.random.seed(seed)

# running for longet period of time to simulate > 14 days
for _ in range(20):
    for i in d: # number of requests per time interval i.e. 30 seconds
        total_time = 30
        try:
            # average inter arrival time in seconds (lambda)
            average_inter_arrival_time = total_time/i
            # generate inter arrival times i.e. at what seconds they will arrive 
            times = np.random.poisson(lam=average_inter_arrival_time, size=i)
            # for every request, generate and wait, if have to
            for k in range(0, i): 
                inter = times[k]
                th = Thread(target=func, args=(1,))
                th.start()
                sleep(inter)
        except Exception as e:
            # i = 0 i.e. no requests during this interval
            average_inter_arrival_time = 0
            times = [0]
        times = sum(times)
        if ((times) < 30):
            sleep(30 - times)
