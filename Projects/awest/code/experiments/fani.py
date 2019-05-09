
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing.dummy import Pool

shape = (30, 2, 10)
array = np.random.random(shape)

fig = plt.figure()
func = lambda i: plt.plot(array[i,0], array[i,1], '.', color='k')
frames = shape[0]
init_func = lambda : None
ani = FuncAnimation(fig, func, frames, init_func)
plt.show()
