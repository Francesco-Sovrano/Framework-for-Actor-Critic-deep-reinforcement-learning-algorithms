import math
import numpy as np
import operator
from scipy import optimize

# get command line args
import options
flags = options.get()

from bisect import bisect_right
from collections import deque
import matplotlib
matplotlib.use('Agg') # non-interactive back-end
from matplotlib import pyplot as plt

from environment.environment import Environment
	
class CarControllerEnvironment(Environment):

	def get_state_shape(self):
		return (1,6,2)

	def get_action_shape(self):
		return (1,) # steering angle, continuous control
	
	def __init__(self, thread_index):
		Environment.__init__(self)
		self.thread_index = thread_index
		self.max_distance_to_path = 0.1 # meters
		self.max_noise_diameter = 0.4 # diameter in meters
		self.speed = 0.1 # meters per step (assuming a step each 0.1 seconds -> 3.6km/h)
		self.max_angle_degrees = 30
		self.max_angle_radians = convert_degree_to_radians(self.max_angle_degrees)
		self.positions_number = 100
		self.max_step = self.positions_number
		self.noise_per_step = self.max_noise_diameter/self.max_step
		# evaluator stuff
		self.episodes = deque()
		
	def build_random_path(self):
		# setup environment
		self.U1, self.V1 = generate_random_polynomial()
		self.U2, self.V2 = generate_random_polynomial()
		# we generate all points for both polynomials, then we shall draw only a portion of them
		self.positions = np.linspace(start=0, stop=1, num=self.positions_number)
		theta = angle(1, self.U1, self.V1)
		xv,yv = poly(self.positions,self.U1), poly(self.positions,self.V1)
		xv2,yv2 = rotate_and_shift(poly(self.positions,self.U2),poly(self.positions,self.V2),xv[-1],yv[-1],theta)
		x = np.concatenate((xv,xv2),axis=0) # length 200
		y = np.concatenate((yv,yv2),axis=0) # length 200
		return x, y
		
	def is_terminal_position(self, position):
		return position >= self.positions_number
		
	def reset(self):
		self.path = self.build_random_path()
		self.noisy_point = self.car_point = (0, 0) # car point and orientation are always expressed with respect to the initial point and orientation of the road fragment
		self.noisy_progress = self.car_progress = 0
		self.noisy_waypoint = self.car_waypoint = 1
		
		self.cumulative_reward = 0
		self.step = 0
		self.last_action = 0
		self.last_reward = 0
		self.last_state = self.get_state(self.car_point, self.car_waypoint)
		
	def get_car_position_and_next_waypoint(self, car_point):
		car_x, car_y = car_point
		def get_distance(position):
			x, y = poly(position,self.U1), poly(position,self.V1)
			return euclidean_distance(car_point,(x,y))
		result = optimize.minimize_scalar(get_distance, bounds=(0,1)) # find the closest spline point
		car_position = result.x
		# Find leftmost value greater than x: the next_waypoint
		next_waypoint = bisect_right(self.positions, car_position)
		# print(car_position, next_waypoint)
		return car_position, next_waypoint
		
	def compute_new_car_point(self, car_point, next_waypoint, compensation_angle):
		# get baseline angle
		xs, ys = self.path
		relative_path, _ = get_relative_path_and_next_position_id(point=car_point, path=([xs[next_waypoint]], [ys[next_waypoint]]))
		_, yc = relative_path		
		baseline_angle = angle(self.positions[next_waypoint], self.U1, self.V1) + np.arctan(yc[0]/(self.speed*10)) # use the tangent to the next waypoint as default direction -> more stable results
		car_angle = baseline_angle + compensation_angle # adjust the default direction using the compensation_angle
		# change point
		car_x, car_y = car_point
		car_x += self.speed*np.cos(car_angle)
		car_y += self.speed*np.sin(car_angle)
		return (car_x, car_y)
		
	def get_noisy_point(self, point):
		x, y = point
		# add noise to point
		x += (2*np.random.random()-1)*self.noise_per_step
		y += (2*np.random.random()-1)*self.noise_per_step
		return (x, y)

	def process(self, policy):
		policy_choice=0
		action = np.clip(policy[0],0,1)
		# get compensation angle
		compensation_angle = (2*action-1)*self.max_angle_radians
		# update perceived car point
		self.car_point = self.compute_new_car_point(self.car_point, self.car_waypoint, compensation_angle)
		# update real car point
		self.noisy_point = self.get_noisy_point(self.compute_new_car_point(self.noisy_point, self.car_waypoint, compensation_angle)) # use perceived waypoint here!
		# update position and direction
		car_position, car_waypoint = self.get_car_position_and_next_waypoint(self.car_point)
		noisy_position, noisy_waypoint = self.get_car_position_and_next_waypoint(self.noisy_point)
		# compute real reward
		noisy_reward = self.get_reward(self.noisy_point, self.noisy_progress, noisy_position)
		if noisy_position > self.noisy_progress: # is moving toward next position
			self.noisy_progress = noisy_position # progress update
			self.noisy_waypoint = noisy_waypoint
		# compute perceived reward
		car_reward = self.get_reward(self.car_point, self.car_progress, car_position)
		if car_position > self.car_progress: # is moving toward next position
			self.car_progress = car_position # progress update
			self.car_waypoint = car_waypoint
		# compute state (after updating progress)
		state = self.get_state(self.car_point, self.car_waypoint)
		# update last action/state/reward
		self.last_state = state
		self.last_action = action
		self.last_reward = car_reward
		# update cumulative reward
		self.cumulative_reward += noisy_reward
		# update step
		self.step += 1
		terminal = self.is_terminal_position(self.car_waypoint) or self.is_terminal_position(self.noisy_waypoint) or self.step >= self.max_step
		if terminal: # populate statistics
			self.episodes.append( {"reward":self.cumulative_reward, "step": self.step, "completed": 1 if self.is_terminal_position(self.car_waypoint) else 0} )
			if len(self.episodes) > flags.match_count_for_evaluation:
				self.episodes.popleft()
		return policy_choice, state, noisy_reward, terminal
		
	def get_concatenation(self):
		return [self.last_action, self.last_reward]
		
	def get_reward(self, car_point, car_progress, car_position):
		if car_position > car_progress: # is moving toward next position
			x, y = poly(car_position,self.U1), poly(car_position,self.V1)
			distance = euclidean_distance(car_point,(x,y))
			# distance = get_distance(self.positions[next_waypoint]) if next_waypoint < self.positions_number else get_distance(1)
			return 1 - np.clip(distance/self.max_distance_to_path,0,1) # always in [0,1] # smaller distances to path give higher rewards
		return -0.1 # is NOT moving toward next position
		
	def get_state(self, car_point, next_waypoint):
		if next_waypoint >= self.positions_number:
			next_waypoint = self.positions_number - 1
		xs, ys = self.path
		next_x, next_y = xs[next_waypoint], ys[next_waypoint]
		car_x, car_y = car_point
		shape = self.get_state_shape()
		state = np.zeros(shape)
		state[0][0] = [car_x, car_y]
		state[0][1] = [next_x, next_y]
		for i in range(len(self.U1)):
			state[0][2+i] = [self.U1[i],self.V1[i]]
		return state
		
	def get_screen(self):
		relative_path, position_id = get_relative_path_and_next_waypoint(point=self.car_point, path=self.path)
		xc, yc = relative_path
		xct = xc[position_id:min(position_id+self.positions_number,2*self.positions_number)] if position_id < len(xc) else []
		yct = yc[position_id:min(position_id+self.positions_number,2*self.positions_number)] if position_id < len(xc) else []
		return (xct,yct)
		
	def get_frame_info(self, network, observation, policy, value, action, reward):
		state_info = "reward={}, action={}, agent={}, value={}\n".format(reward, action, network.agent_id, value)
		policy_info = "policy={}\n".format(policy)
		frame_info = { "log": state_info + policy_info }
		if flags.save_episode_screen:
			# First set up the figure, the axis, and the plot element we want to animate
			fig, ax = plt.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10))
			line, = ax.plot(observation[0], observation[1], lw=2)
			ax.plot(0,0,"ro")
			fig.canvas.draw()
			# Now we can save it to a numpy array.
			data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
			data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			plt.close(fig) # release memory
			frame_info["screen"] = { "value": data, "type": 'RGB' }
		return frame_info
		
	def get_statistics(self):
		result = {}
		result["avg_reward"] = 0
		result["avg_step"] = 0
		result["avg_completed"] = 0
		count = len(self.episodes)
		if count>0:
			for e in self.episodes:
				result["avg_reward"] += e["reward"]
				result["avg_step"] += e["step"]
				result["avg_completed"] += e["completed"]
			result["avg_reward"] /= count
			result["avg_step"] /= count
			result["avg_completed"] /= count
		return result
		
def rot(x,y,theta):
	#rotation and translation
	xnew = x*np.cos(theta)- y*np.sin (theta)
	ynew = x*np.sin(theta) + y*np.cos(theta) 
	return(xnew,ynew)

def shift_and_rotate(xv,yv,dx,dy,theta):
	#xv and yv are lists
	# xy = zip(xv,yv)
	# xynew = [rot(c[0]+dx,c[1]+dy,theta) for c in xy]
	# xyunzip = list(zip(*xynew))
	# return(xyunzip[0],xyunzip[1])
	nex_xv = []
	nex_yv = []
	for i in range(len(xv)):
		x, y = rot(xv[i]+dx,yv[i]+dy,theta)
		nex_xv.append(x)
		nex_yv.append(y)
	return nex_xv, nex_yv

def rotate_and_shift(xv,yv,dx,dy,theta):
	#xv and yv are lists
	# xy = zip(xv,yv)
	# xynew = [tuple(map(operator.add,rot(c[0],c[1],theta),(dx,dy))) for c in xy]
	# xyunzip = list(zip(*xynew))
	# return(xyunzip[0],xyunzip[1])
	nex_xv = []
	nex_yv = []
	for i in range(len(xv)):
		x, y = rot(xv[i],yv[i],theta)
		nex_xv.append(x + dx)
		nex_yv.append(y + dy)
	return nex_xv, nex_yv

def generate_random_polynomial():
	#both x and y are defined by two polynomials in a third variable p, plus
	#an initial angle (that, when connecting splines, will be the same as
	#the final angle of the previous polynomial)
	#Both polynomials are third order.
	#The polynomial for x is aU, bU, cU, dU
	#The polynomial for y is aV, bV, cV, dV
	#aU and bU are always 0 (start at origin) and bV is always 0 (derivative at
	#origin is 0). bU must be positive
	# constraints initial coordinates must be the same as
	# ending coordinates of the previous polynomial
	aU = 0
	aV = 0
	# initial derivative must the same as the ending
	# derivative of the previous polynomial
	bU = (10-6)*np.random.random()+6  #around 8
	bV = 0
	#we randonmly generate values for cU and dU in the range ]-1,1[
	cU = 2*np.random.random()-1
	dU = 2*np.random.random()-1
	finalV = 10*np.random.random()-5
	#final derivative between -pi/6 and pi/6
	finald = np.tan((np.pi/3)*np.random.random() - np.pi/6)
	#now we fix parameters to meet the constraints:
	#bV + cV + dV = finalV 
	#angle(1) = finald; see the definition of angle below
	Ud = bU + 2*cU + 3*dU
	#Vd = bU + 2*cU + 3*dU = finald*Ud
	dV = finald*Ud - 2*finalV + bV
	cV = finalV - dV - bV
	return ((aU,bU,cU,dU), (aV,bV,cV,dV))

def poly(p, points):
	(a,b,c,d) = points
	p_squared = p**2 # optimization
	return a + b*p + c*p_squared + d*p_squared*p

def derivative(p, points):
	(_,b,c,d) = points
	return b + 2*c*p + 3*d*p**2

def angle(p, U, V):
	Ud = derivative(p,U)
	Vd = derivative(p,V)
	return (np.arctan(Vd/Ud)) if abs(Ud) > abs(Vd/1000) else (np.pi/2)
	
def norm(angle):
    if angle >= np.pi:
        angle -= 2*np.pi
    elif angle < -np.pi:
        angle += 2*np.pi
    return angle

def convert_degree_to_radians(degree):
	return (degree/180)*np.pi
	
def get_relative_path_and_next_waypoint(point, path):
	point_x, point_y = point
	path_xs, path_ys = path
	relative_path = shift_and_rotate(path_xs, path_ys, -point_x, -point_y, 0)
	xc, yc = relative_path
	next_waypoint = 0
	while next_waypoint < len(xc) and xc[next_waypoint] < 0:
		next_waypoint +=1
	return relative_path, next_waypoint
	
def euclidean_distance(a,b):
	sum = 0
	for i in range(len(a)):
		sum += (a[i]-b[i])**2
	return math.sqrt(sum)