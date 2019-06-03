# Exercise File
from data_assimilation.ParticleFilter import ParticleFilter
# from jupyterthemes.jtplot import style as jtstyle
# jtstyle('gruvboxd')


if 0:  # screensaver

	from models.screensaver import Model
	model = Model()
	pf = ParticleFilter(model, particles=10, window=1, do_copies=False, do_save=True)
	pf.batch(self, model, iterations=11, do_ani=False, agents=None)

if 1:  # sspmm

	from models.sspmm import Model
	model = Model()

	if 0:  # Test Model
		model.batch()
	else:  # Test PF
		pf = ParticleFilter(model, particles=100, window=10, do_copies=False, do_save=True)
		pf.batch(model, do_ani=True, agents=1)
