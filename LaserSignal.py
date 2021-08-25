import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def signal(start):

    strength = 0.0
    end = 50
    t = np.linspace(0,end)
    point_number = len(t)
    y = []
    current = start
    
    for i in range(point_number):
        y.append(current + 0.5*(np.random.random()-0.5))
        current = y[i]
    
    y = interp1d(t, y, kind='cubic')
    t = np.linspace(0, end, num=end, endpoint=True)
    y = y(t) + strength*np.random.random(len(y(t)))-0.5 + (0.1*np.sin(t/75)-0.5)
    
        
    return current,y,t

def generateSignal(repeats):
    current = 0
    output = []
    final_output = []
    
    for i in range(repeats):  
        current,y,t= signal(current)
        output.append(y)
        
    for i in range(len(output)):
        for j in range(len(output[0])):
            final_output.append(output[i][j])
    return final_output

