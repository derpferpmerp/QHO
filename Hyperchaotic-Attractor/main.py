from lode import System
import numpy as np


class Hyper(System):
	def __init__(self,
		alpha=-2,
		beta=1,
		gamma=0.2,
		eta=1,
		state0=[0.5, 0.5, 0.5, 0.5]
	):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.eta = eta
		self.state0 = state0
		self.fig = None
		self.axes = None
	
	def derive(self, state, list_t):
		x, y, z, w = state
		return (
			x * (1 - y) + (self.alpha * z),
			self.beta * (pow(x,2)-1) * y,
			self.gamma * (1 - y) * w,
			self.eta * z,
		)

	def calculateSmart(self, list_t, cross=2):
		SYSTEM.calculateSystem(list_t)
		statesWZ, statesXZ, statesXW = [[],[],[]]
		for i_y, y in enumerate(self.states[:, 1]):
			if y == cross or cross is False:
				X, Y, Z, W = self.states[i_y]
				statesWZ.append([W, Z])
				statesXZ.append([X, Z])
				statesXW.append([X, W])
		data=[ statesWZ, statesXZ, statesXW ]
		for i, statesArray in enumerate(data):
			self.states = np.array(statesArray)
			SYSTEM.graph(
				outfile="hyper.png",
				hideAxis=False,
				dimensions=2,
				ax_indx=i,
				subplots=(1, 3),
				saveFile=bool(i == len(data) - 1)
			)
		
SYSTEM = Hyper()
T = np.arange(0.0, 100.0, 0.01)
SYSTEM.calculateSmart(T, cross=False)
