

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig = plt.figure()
frames = 100
data = np.random.random((frames,2,10))
sc = plt.scatter(data[0,0], data[0,1], s=100)
func = lambda i: sc.set_array(data[i])
ani = animation.FuncAnimation(fig, func, frames)
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# shape = (2, 30, 20)
# data = np.random.rand(shape)
#
# fig = plt.figure(figsize=(8,4.5))
#
# # plt.axes([0, 2*np.pi], [-1, +1])
#
# x = np.linspace(0, 2*np.pi, 300)
#
# ln, = ax.plot(x, )
# func = lambda i: ln.set_data(locs[i,0], locs[i,1])
# frames = np.linspace(0, 2*np.pi, 100)
# init_func = lambda: ln.set_ydata(np.nan*len(x))
#
#
#
# ani = animation.FuncAnimation(fig, func, frames, init_func)
# plt.pause(4)
# # ani.save("movie.mp4")
# # ani.to_html5_video()
# # ani.jshtml()
