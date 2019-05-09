# Particle Filter
'''
A Particle Filter design for Agent-Based Modelling
v7.3 (lit)
'''
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from copy import deepcopy
from multiprocessing.dummy import Pool
np.random.seed(39)


class ParticleFilter:

	def __init__(self, model0, particles=10, window=1, do_copies=True, do_save=False, do_noise=False, do_paral=True):
		self.time = 0
		# Params
		self.window = window
		self.particles = particles
		# Model
		self.models = [deepcopy(model0) for _ in range(self.particles)]
		if not do_copies:
			[model.__init__(model0.params) for model in self.models]
		for unique_id in range(self.particles):
			self.models[unique_id].unique_id = unique_id
		# Save
		self.do_save = do_save
		self.do_noise = do_noise
		self.do_paral = do_paral
		if self.do_save:
			self.active = []
			self.mean = []
			self.var = []
			self.err = []
			self.resampled = []
		return

	def step(self, state_obs):
		self.time += self.window
		states = np.array([model.get_state() for model in self.models])
		states = self.predict(states)
		weights = self.reweight(states, state_obs)
		states, weights = self.resample(states, weights)
		if self.do_save: self.save(states, weights, state_obs)
		return

	def predict(self, states):
		[self.models[i].set_state(states[i]) for i in range(self.particles)]
		if self.do_paral:
			Pool().map(lambda model: [model.step() for _ in range(self.window)], self.models)
		else:
			[map(lambda model: [model.step() for _ in range(self.window)], self.models)]
		states = np.array([model.get_state() for model in self.models])
		return states

	def reweight(self, states, state_obs):
		if states.shape[1] != state_obs.shape[0]:
			print('Warning - Not equal states {} and state_obs {} lengths.\nShortening quick fix applied.'.format(states.shape,state_obs.shape))
			states = states[:, :len(state_obs)]
		distance = np.linalg.norm(states - state_obs, axis=1)
		weights = 1 / np.fmax(1e-99, distance)
		weights /= np.sum(weights)
		return weights

	def resample(self, states, weights):
		states, weights, indexes = self.resample_systematic(states, weights)
		# Add resample noise
		if self.do_noise:
			std = np.std(states, 0)
			conditions = indexes-np.arange(len(indexes))
			conditions = abs(conditions[1:] - conditions[:-1])
			noise = np.zeros(states.shape)
			for i, condition in enumerate(conditions):
				if condition:
					noise[i] = np.random.normal(0, std)
			states = states + noise  # += does not work with numpy arrays
		if self.do_save:
			self.resampled.append(len(indexes) - len(set(indexes)))
		return states, weights

	def resample_systematic(self, states, weights):
		offset = (np.arange(self.particles) + np.random.uniform()) / self.particles
		cumsum = np.cumsum(weights)
		i, j = 0, 0
		indexes = np.empty(self.particles, 'i')
		while i < self.particles and j < self.particles:
			if offset[i] < cumsum[j]:
				indexes[i] = j
				i += 1
			else:
				j += 1
		states = states[indexes]
		weights = weights[indexes]
		return states, weights, indexes

	def resample_stratified(self, states, weights):
		N = len(weights)
		offset = (np.random.rand(N) + range(N)) /N
		cumsum = np.cumsum(weights)
		i, j = 0, 0
		indexes = np.empty(self.particles, 'i')
		while i < self.particles and j < self.particles:
			if offset[i] < cumsum[j]:
				indexes[i] = j
				i += 1
			else:
				j += 1
		states = states[indexes]
		weights = weights[indexes]
		return states, weights, indexes

	def save(self, states, weights, state_obs):
		mean = np.average(states, axis=0)
		self.mean.append(np.average(mean))
		var = np.average((states - mean)**2, weights=weights, axis=0)
		self.var.append(np.average(var))
		err = np.linalg.norm(mean - state_obs)
		self.err.append(err)
		return

	def file(self, do_file=True):
		if self.do_save:
			mean = np.array(self.mean, 'f')
			var = np.array(self.var, 'f')
			if np.any(var) < 0: print('Warning - Negative variance')
			if np.any(var) == np.nan: print('Warning - A NaN variance')
			err = np.array(self.err, 'f')
			if np.any(err) < 0: print('Warning - Negative error')
			if np.any(err) == np.nan: print('Warning - A NaN variance')
			if not do_file:
				return mean, var, err
			else:
				np.savez('pf_data', mean, var, err)
		else:
			print('Warning - Cannot file as do_save is: ', self.do_save)
		return

	def plot(self, do_plot=True):
		if self.do_save:
			mean, var, err = self.file(False)

			# Expectation
			plt.figure()
			plt.plot(mean)
			plt.xlabel('step id')
			plt.ylabel('mean')

			# Error
			plt.figure()
			plt.plot(err)
			plt.fill_between(range(len(err)), err-var, err+var, alpha=.5)
			plt.xlabel('step id')
			plt.ylabel('error')

			# Resampling Analytics
			n = 3
			cumsum = np.cumsum(self.resampled)
			smoothed = (cumsum[n:]-cumsum[:-n])/n
			plt.figure()
			plt.plot(self.resampled, label='resampled')
			plt.plot(smoothed, label='smoothed, 3')
			plt.xlabel('step id')
			plt.ylabel('resampled')
			plt.legend()
			plt.show()
		else:
			print('Warning - Cannot do_plot as do_save is: ', self.do_save)
		return

	def ani(self, model0, agents=None):
		if self.time%10 is 0:
			plt.clf()
			fig = plt.figure(1)
			[model.ani(agents=agents, colour='r', alpha=.3) for model in self.models]
			model0.ani(agents=None,   colour='b', alpha=.6)
			model0.ani(agents=agents, show_separation=True)
			plt.pause(1/4)
			# plt.savefig('./data/run2/sspmm_pf_{}.png'.format(self.time))
		return

	def batch(self, model0, iterations=None, do_ani=False, agents=None):
		if iterations is None:
			iterations = model0.iterations
		for _ in range(iterations):
			[model0.step() for _ in range(self.window)]
			state_obs = model0.get_state()
			self.step(state_obs)
			if do_ani:
				self.ani(model0, agents)
		if self.do_save:
			self.plot()
			# self.file()
		return

if __name__ == '__main__':
	from sspmm import Model
	model = Model({'pop_total':10})
	pf = ParticleFilter(model, particles=20, window=10, do_copies=False, do_save=True, do_noise=False, do_paral=True)
	pf.batch(model, iterations=10, do_ani=True, agents=1)
