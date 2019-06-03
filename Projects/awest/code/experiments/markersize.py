# A figure where the markersize is scaled to the grid
import matplotlib.pyplot as plt

width = 45
height = 20

wid = 8
hei = wid / width * height

scale = 5  # dot radius
markersize = scale * 3*72 * hei / height  # 72px/in

data = ([5,15,25,35,45], [10,5,15,5,10])

fig = plt.figure(figsize=(wid, hei))
plt.plot(*data, marker='.', markersize=markersize)
plt.axis([0, width, 0, height], aspect='equal')
# plt.tight_layout(pad=0)
plt.show()
