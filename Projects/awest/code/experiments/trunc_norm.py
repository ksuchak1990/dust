# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.empty(100_000)
# i = 0
# while i < len(x):
# 	r = -2
# 	while r <= -2:
# 		r = np.random.normal()
# 	x[i] = r
# 	i += 1
#
# y = np.fmax(-2, np.random.normal(size=100_000))
#
# plt.hist(x, bins=100, alpha=.5)
# plt.hist(y, bins=100, alpha=.5)
# plt.show()


from matplotlib import pyplot as plt
import numpy as np
from scipy.special import erf

def tnorm(x, mu=0, sig=1, lower=None, upper=None):
	if lower is not None and upper is not None:
		lower, upper = min(lower, upper), max(lower, upper)
	phi = lambda z: np.exp(-z*z/2) / np.sqrt(2*np.pi)
	Phi = lambda x: (1 +erf(x/np.sqrt(2))) / 2
	y = phi((x-mu)/sig)
	if lower is not None:
		y[lower>x] = 0
		lower = Phi((lower-mu)/sig)
	else:
		lower = 0
	if upper is not None:
		y[x>upper] = 0
		upper = Phi((upper-mu)/sig)
	else:
		upper = 1
	y /= sig*(upper-lower)
	# 1 ≈ sum(y*h), y=(y[1:]+y[:-1])/2, h=x[1:]-x[:-1]
	return y


min = .1
x = np.linspace(0, 5, 1000)
y = tnorm(x, mu=1, sig=1, lower=min)
plt.figure(figsize=(8,4.5))
plt.plot(x, y)
plt.axis('off')
plt.show()
