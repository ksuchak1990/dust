# Mesafree 2

class Model(object):
	step = {
		'': None,
		'id': 0,
		'limit': 1800,
	}
	def __init__(self, unique_id=None):
		self.unique_id = unique_id
		print(self.step.id)

	def _step(self):
		pass

Model()
