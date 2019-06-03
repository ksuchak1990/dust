A small experiment on which interpolation technique returns the best results for the heatmap.
```python
def heatmap(data, int):
	hdata, *bins = np.histogram2d(*data, (self.width/10, self.height/10))
	plt.figure(figsize=figsize, dpi=dpi)
	plt.axis(np.ravel(self.boundaries,'f'))
	plt.axis('off')
	plt.tight_layout(pad=0)
	plt.imshow(hdata.T, interpolation=int, extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]), cmap='gray_r')
	return
```
Check `get_plot` for data usage.
```
for int in ['none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']:
	heatmap(data, int)
```
