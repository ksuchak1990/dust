# sspmm
'''
A genuine Agent-Based Model designed to contain many ABM features.
v7.4

New:
	get_ani  - saveable animation
	activation  - entrance limitors instead of entrance time
	get_plot  - added new heatmap

TODO:
	hboxplot for parametrics
	replace kdeplot


'''
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import seaborn as sns

def update_dict(dict0, dict1):
	dict2 = dict()
	for key, value in dict1.items():
		if key in dict0:
			if dict0[key] is not dict1[key]:
				dict0[key] = dict1[key]
				if 'do_' not in key:
					dict2[key] = dict1[key]
		else:
			print(f'BadKeyWarning: {key} is not a model parameter.')
	return dict0, dict2


class Agent:

	def __init__(self, model, unique_id):
		self.unique_id = unique_id
		# Required
		self.status = 0  # 0 Not Started, 1 Active, 2 Finished
		# Location
		self.entrance = np.random.randint(model.entrances)
		self.loc_start = model.loc_entrances[self.entrance]
		self.loc_start[1] += model.entrance_space * np.random.uniform(-1,+1)
		self.exit = np.random.randint(model.exits)
		self.loc_desire = model.loc_exits[self.exit]
		self.location = self.loc_start
		# Parameters
		self.speed_max = 0
		while self.speed_max <= model.speed_min:
			self.speed_max = np.random.normal(model.speed_mean, model.speed_std)
		self.wiggle = min(model.max_wiggle, self.speed_max)
		self.speeds = np.arange(self.speed_max, model.speed_min, -model.speed_step)
		if model.do_save:
			self.wiggles = 0  # number of wiggles this agent has experienced
			self.wiggle_map = []
			self.collisions = 0  # number of speed limitations/collisions this agent has experienced
			self.history_loc = []
		return

	def step(self, model):
		if self.status == 0:
			self.activate(model)
		elif self.status == 1:
			self.move(model)
			self.exit_query(model)
		self.save(model)
		return

	def activate(self, model):
		if model.entrance_current[self.entrance] < model.entrance_speed:
			model.entrance_current[self.entrance] += 1
			self.status = 1
			model.pop_active += 1
			self.time_start = model.time
		return

	@staticmethod
	def distance(loc1, loc2):
		# Euclidean distance between two 2D points.
		# norm = np.linalg.norm(loc1-loc2)  # 2.45s
		x = loc1[0] - loc2[0]
		y = loc1[1] - loc2[1]
		norm = (x*x + y*y)**.5  # 1.71s
		return norm

	def move(self, model):
		direction = (self.loc_desire - self.location) / self.distance(self.loc_desire, self.location)
		for speed in self.speeds:
			# Direct
			new_location = self.location + speed * direction
			if not self.collision(model, new_location):
				break
			else:
				if model.do_save:
					self.collisions += 1
					model.collision_map.append(self.location)
			# Wiggle
			if speed == self.speeds[-1]:
				if model.do_save:
					self.wiggles += 1
					model.wiggle_map.append(self.location)
				new_location = self.location + [0, self.wiggle*np.random.randint(-1, 1+1)]
				if not model.is_within_bounds(new_location):
					new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
		# Move
		self.location = new_location
		return

	def collision(self, model, new_location):
		if not model.is_within_bounds(new_location):
			collide = True
		elif self.neighbourhood(model, new_location):
			collide = True
		else:
			collide = False
		return collide

	def neighbourhood(self, model, new_location):
		neighbours = False
		neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
		for neighbouring_agent in neighbouring_agents:
			agent = model.agents[neighbouring_agent]
			if agent.status == 1 and self.unique_id != agent.unique_id and new_location[0] <= agent.location[0]:
				neighbours = True
				break
		return neighbours

	def exit_query(self, model):
		# if model.width-self.location[0] < model.exit_space:  # saves a small amount of time
		if self.distance(self.location, self.loc_desire) < model.exit_space:
			self.status = 2
			model.pop_active -= 1
			model.pop_finished += 1
			if model.do_save:
				time_exp = (self.distance(self.loc_start, self.loc_desire) - model.exit_space) / self.speed_max
				model.time_exped.append(time_exp)
				time_taken = model.time - self.time_start
				model.time_taken.append(time_taken)
				time_delay = time_taken - time_exp
				model.time_delay.append(time_delay)
		return

	def save(self, model):
		if model.do_save:
			if self.status==1:
				self.history_loc.append(self.location)
			else:
				self.history_loc.append((None, None))
		return


class Model:

	def __init__(self, unique_id=None, **kwargs):
		self.unique_id = unique_id
		self.id = time.strftime('%y%m%d_%H%M%S')
		# Params
		params = {
			'width': 400,
			'height': 200,
			'pop_total': 100,

			'entrances': 3,
			'exits': 2,
			'entrance_space': 2,
			'exit_space': 2,
			'entrance_speed': .1,

			'speed_min': .2,
			'speed_mean': 1,
			'speed_std': 1,
			'speed_steps': 3,

			'separation': 5,
			'max_wiggle': 1,

			'iterations': 3600,

			'do_save': False,
			'do_print': True,
			'do_plot': False,
			'do_ani': False,
			'do_file': False
			}
		self.params, self.params0 = update_dict(params, kwargs)
		[setattr(self, key, value) for key, value in self.params.items()]
		# Constants
		self.speed_step = (self.speed_mean - self.speed_min) / self.speed_steps
		self.boundaries = np.array([[0, 0], [self.width, self.height]])
		init_gates = lambda x,y,n: np.array([np.full(n,x), np.linspace(0,y,n+2)[1:-1]]).T
		self.loc_entrances = init_gates(0, self.height, self.entrances)
		self.loc_exits = init_gates(self.width, self.height, self.exits)
		self.entrance_current = np.zeros(self.entrances)
		# self.exit_flow = np.zeros(self.exits)
		# Variables
		self.time = 0
		self.pop_active = 0
		self.pop_finished = 0
		self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
		self.tree = None
		if self.do_save:
			self.time_exped = []
			self.time_taken = []
			self.time_delay = []
			self.collision_map = []
			self.wiggle_map = []
		self.is_within_bounds = lambda loc: all(self.boundaries[0] <= loc) and all(loc <= self.boundaries[1])
		return

	def step(self):
		self.entrance_current[:] -= self.entrance_speed
		if self.pop_finished < self.pop_total:
			self.tree = cKDTree([agent.location for agent in self.agents])
			[agent.step(self) for agent in self.agents]
		self.time += 1
		return

	def get_state(self, sensor='location'):
		if sensor is None:
			state = [(agent.status, *agent.location) for agent in self.agents]
			state = np.append(self.time, np.ravel(state))
		elif sensor is 'location':
			state = [agent.location for agent in self.agents]
			state = np.ravel(state)
		return state

	def set_state(self, state, sensor='location', noise=False):
		if sensor is None:
			self.time = int(state[0])
			state = np.reshape(state[1:], (self.pop_total, 3))
			for i, agent in enumerate(self.agents):
				agent.status = int(state[i,0])
				agent.location = state[i,1:]
		elif sensor is 'location':
			state = np.reshape(state, (self.pop_total, 2))
			for i, agent in enumerate(self.agents):
				agent.location = state[i,:]
		return

	def batch(self):
		for i in range(self.iterations):
			self.step()
			if self.pop_finished == self.pop_total:
				if self.do_print:
					print('Everyone made it!')
				break
		if self.do_save:
			if self.do_print:
				print(self.get_stats())
			if self.do_plot:
				figs = self.get_plot()
			if self.do_ani:
				ani = self.get_ani(show_separation=True)
			if not self.do_file:
				plt.show()
		if True:#self.do_pickle:
			pass
		return

	def get_stats(self):
		statistics = {
			'Finish Time': self.time,
			'Total': self.pop_total,
			'Active': self.pop_active,
			'Finished': self.pop_finished,
			'Time Delay': np.mean(self.time_delay),
			'Interactions': np.mean([agent.collisions for agent in self.agents]),
			'Wiggles': np.mean([agent.wiggles for agent in self.agents]),
			'GateWiggles': sum(wig[0]<self.entrance_space for wig in self.wiggle_map)/self.pop_total,
			}
		if 1:
			pround = lambda x,p: float(f'%.{p-1}e'%x)
			for k,v in statistics.items():
				statistics[k] = pround(v,4)
		if self.do_file:
			print(statistics, file=open(f'{self.id}_stats.txt','w'))
		return statistics

	def get_plot(self):
		wid = 8
		rel = wid / self.width
		hei = rel * self.height
		figsize = (wid,hei)
		dpi = 160
		figs = []

		if 1:  # Trails
			fig0 = plt.figure(figsize=figsize, dpi=dpi)
			plt.axis(np.ravel(self.boundaries,'f'))
			plt.axis('off')
			plt.plot([], 'b')
			plt.plot([], 'g')
			plt.title('Agent Trails')
			plt.legend(['Active', 'Finished'])
			plt.tight_layout(pad=0)
			for agent in self.agents:
				if agent.status == 1:
					alpha = 1
					colour = 'b'
				elif agent.status == 2:
					alpha = .5
					colour = 'g'
				else:
					alpha = 1
					colour = 'r'
				locs = np.array(agent.history_loc).T
				plt.plot(*locs, color=colour, alpha=alpha, linewidth=.5)
			if self.do_file:
				plt.savefig(f'{self.id}_trails.png')
			figs.append(fig0)

		if 1:  # Time Expected/Taken/Delay Histogram
			fig1 = plt.figure(figsize=figsize, dpi=dpi)
			fmax = max(np.amax(self.time_exped), np.amax(self.time_taken), np.amax(self.time_delay))
			sround = lambda x,p: float(f'%.{p-1}e'%x)
			bins = np.linspace(0, sround(fmax,2), 20)
			plt.hist(self.time_exped, bins=bins+4, alpha=.5, label='Expected')
			plt.hist(self.time_taken, bins=bins+2, alpha=.5, label='Taken')
			plt.hist(self.time_delay, bins=bins+0, alpha=.5, label='Delayed')
			plt.xlabel('Time')
			plt.ylabel('Number of Agents')
			plt.grid(False)
			plt.legend()
			plt.tight_layout(pad=0)
			if self.do_file:
				plt.savefig(f'{self.id}_timehist.png')
			figs.append(fig1)

		def heightmap(data, ax=None, kdeplot=True, cmap=None, alpha=.7):
			if kdeplot:
				sns.kdeplot(*data, ax=ax, cmap=cmap, alpha=alpha, shade=True, shade_lowest=False)
			else:
				hdata, *bins = np.histogram2d(*data, (20, 10))
				ax.contourf(hdata, cmap=cmap, alpha=alpha, extend='min', extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]))
			return ax

		if 1:  # Wiggle heightmap
			fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
			fig.tight_layout(pad=0)
			heightmap(np.array(self.collision_map).T, ax=ax)
			heightmap(np.array(self.wiggle_map).T, ax=ax)
			ax.set(frame_on=False, aspect='equal', xlim=self.boundaries[:,0], xticks=[], ylim=self.boundaries[:,1], yticks=[])
			ax.legend()
			if self.do_file:
				plt.savefig(f'{self.id}_mapints.png')
			figs.append(fig)

		if 1:  # Path heightmap
			history_locs = []
			for agent in self.agents:
				for loc in agent.history_loc:
					if None not in loc:
						history_locs.append(loc)
			history_locs = np.array(history_locs).T
			fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
			fig.tight_layout(pad=0)
			heightmap(history_locs, ax=ax, cmap='gray_r')
			ax.set(frame_on=False, aspect='equal', xlim=self.boundaries[:,0], xticks=[], ylim=self.boundaries[:,1], yticks=[])
			if self.do_file:
				plt.savefig(f'{self.id}_maplocs.png')
			figs.append(fig)

		return figs

	def get_ani(self, agents=None, colour='k', alpha=.5, show_separation=False, wiggle_map=False):
		# Load Data
		locs = np.array([agent.history_loc for agent in self.agents[:agents]]).transpose((1,2,0))
		wid = 8
		rel = wid / self.width
		hei = rel * self.height
		markersize = self.separation * 216*rel  # 3*72px/in=216
		#
		fig, ax = plt.subplots(figsize=(wid, hei), dpi=160)
		if wiggle_map:
			sns.kdeplot(*np.array(self.collision_map).T, ax=ax, cmap='gray_r', alpha=.3, shade=True, shade_lowest=False)
		ln0, = plt.plot([],[], '.', alpha=.05, color=colour, markersize=markersize)
		ln1, = plt.plot([],[], '.', alpha=alpha, color=colour)
		def init():
			fig.tight_layout(pad=0)
			ax.set(frame_on=False, aspect='equal', xlim=self.boundaries[:,0], xticks=[], ylim=self.boundaries[:,1], yticks=[])
			return ln0, ln1,
		def func(frame):
			if show_separation:
				ln0.set_data(*locs[frame])
			ln1.set_data(*locs[frame])
			return ln0, ln1,
		frames = self.time
		ani = FuncAnimation(fig, func, frames, init, interval=100, blit=True)
		if self.do_file:
			ani.save(f'{self.id}_ani.mp4')
		return ani


# Batches
def parametric_study():
	analytics = {}

	for pop, sep in [(100, 5), (300, 3), (700, 2)]:
		params = {
			'pop_total': pop,
			'separation': sep,
			}
		t = time.time()
		model = Model({**params, 'do_save':True,'do_print':False})
		model.batch()
		analytics[str(model.params0)] = {
			'Process Time': time.time()-t,
			**model.get_stats()
			}
	for s in (9,5,2,1):
		params = {'speed_steps': s}
		t = time.time()
		model = Model({**params, 'do_save':True,'do_print':False})
		model.batch()
		analytics[str(model.params0)] = {
			'Process Time': time.time()-t,
			**model.get_stats()
			}

	csv_str, lines = '', []
	for i,row in enumerate(analytics):
		if i==0:
			header = ', '.join(k for k,_ in analytics[row].items()) + ',\n'
		line = ', '.join(f'{v}' for _,v in analytics[row].items()) + f', {row}'
		lines.append(line)
	csv_str = header + '\n'.join(lines)
	print(csv_str)
	print(csv_str, file=open('test.csv', 'w'))
	return


if __name__ == '__main__':
	# parametric_study()

	np.random.seed(1)
	kwargs = {
		'width': 400,
		'height': 200,
		'pop_total': 100,

		'entrances': 3,
		'exits': 2,
		'entrance_space': 2,
		'exit_space': 2,
		'entrance_speed': .1,

		'speed_min': .3,
		'speed_mean': 1,
		'speed_std': 1,
		'speed_steps': 3,

		'separation': 5,
		'max_wiggle': 1,

		'iterations': 1800,

		'do_save': True,
		'do_print': True,
		'do_plot': False,
		'do_ani': True,
		'do_file': False,
		}
	Model(**kwargs).batch()
