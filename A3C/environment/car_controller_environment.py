import math
import numpy as np
import operator
from scipy import optimize
# import gc

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
		return (1,10,2)

	def get_action_shape(self):
		return (2,) # steering angle, continuous control without softmax
	
	def __init__(self, thread_index):
		Environment.__init__(self)
		self.thread_index = thread_index
		self.max_distance_to_path = 0.1 # meters
		self.max_noise_radius = 0.05 # radius in meters
		self.min_speed = 0.05 # meters per step (assuming a step each 0.1 seconds -> 1.8km/h)
		self.max_speed = 0.2 # meters per step (assuming a step each 0.1 seconds -> 7.2km/h)
		self.max_speed_change_per_step = 0.01
		self.gamma = 10
		self.max_steering_degree = 30
		self.max_compensation_degree = 15
		self.max_steering_angle = convert_degree_to_radiant(self.max_steering_degree)
		self.max_compensation_angle = convert_degree_to_radiant(self.max_compensation_degree)
		self.path_length = 200
		self.max_step = 150
		self.noise_per_step = self.max_noise_radius/self.max_step
		# evaluator stuff
		self.episodes = deque()
	
	def reset(self):
		self.path = self.build_random_path()
		self.noisy_point = self.car_point = (0,0) # car point and orientation are always expressed with respect to the initial point and orientation of the road fragment
		self.car_angle = np.pi/2 + 0.01
		self.noisy_progress = self.car_progress = 0
		self.noisy_waypoint = self.car_waypoint = 1
		
		self.speed = self.min_speed + (self.max_speed-self.min_speed)*np.random.random() # random initial speed in [min_speed,max_speed]
		self.steering_angle = 0
		self.cumulative_reward = 0
		self.step = 0
		self.last_reward = 0
		self.last_state = self.get_state(self.car_point, self.car_waypoint)
		self.avg_speed_per_steps = 0
		
	def get_point_from_position(self, position):
		if position <= 1: # first spline 
			return (poly(position,self.U1), poly(position,self.V1))
		# second spline
		return rotate_and_shift(poly(position-1,self.U2), poly(position-1,self.V2), self.middle_point[0], self.middle_point[1], self.theta)
		
	def get_angle_from_position(self, position):
		if position <= 1: # first spline
			return angle(position, self.U1, self.V1)
		# second spline
		return angle(position-1, self.U2, self.V2)+self.theta
		
	def build_random_path(self):
		# setup environment
		self.U1, self.V1 = generate_random_polynomial()
		self.U2, self.V2 = generate_random_polynomial()
		# we generate all points for both polynomials, then we shall draw only a portion of them
		self.positions = np.linspace(start=0, stop=2, num=self.path_length) # first spline is in [0,1] while the second one is in [1,2]
		self.theta = angle(1, self.U1, self.V1)
		self.middle_point = self.get_point_from_position(1)
		xy = [self.get_point_from_position(pos) for pos in self.positions]
		return list(zip(*xy))

	def is_terminal_position(self, position):
		return position >= self.path_length

	def get_car_position_and_next_waypoint(self, car_point):
		car_x, car_y = car_point
		# Find the closest spline point
		result = optimize.minimize_scalar(lambda pos: euclidean_distance(car_point, self.get_point_from_position(pos)), bounds=(0,2))
		car_position = result.x
		# Find leftmost value greater than x: the next_waypoint
		next_waypoint = bisect_right(self.positions, car_position)
		# print(car_position, next_waypoint)
		return car_position, next_waypoint

	def compute_new_car_point_and_angle(self, car_point, car_angle, steering_angle, speed, add_noise=False):
		# get new car angle
		new_car_angle = car_angle + steering_angle
		# change point
		car_x, car_y = car_point
		car_x += speed*np.cos(new_car_angle)
		car_y += speed*np.sin(new_car_angle)
		if add_noise: # add noise to point
			car_x += (2*np.random.random()-1)*self.noise_per_step
			car_y += (2*np.random.random()-1)*self.noise_per_step
		return (car_x, car_y), new_car_angle

	def compute_new_steering_angle(self, speed, steering_angle, car_point, car_angle, compensation_angle, next_waypoint):
		# get baseline angle
		xs, ys = self.path
		_, yc = shift_and_rotate(xs[next_waypoint], ys[next_waypoint], -car_point[0], -car_point[1], -car_angle) # point relative to car position
		baseline_angle = self.get_angle_from_position(self.positions[next_waypoint]) + np.arctan(yc/(speed*self.gamma)) # use the tangent to the next waypoint as default direction -> more stable results
		# get new angle
		new_angle = compensation_angle + baseline_angle
		# get steering angle
		steering_angle += new_angle - car_angle
		# clip steering angle in [-max_steering_angle, max_steering_angle]
		return np.clip(steering_angle,-self.max_steering_angle,self.max_steering_angle)
		
	def compute_new_speed(self, speed, step_acceleration):
		speed += step_acceleration
		return np.clip(speed, self.min_speed, self.max_speed)

	def process(self, action_vector):
		# compute new speed
		step_acceleration = (2*action_vector[1]-1)*self.max_speed_change_per_step
		self.speed = self.compute_new_speed(speed=self.speed, step_acceleration=step_acceleration)
		# compute new steering_angle
		compensation_angle = (2*action_vector[0]-1)*self.max_compensation_angle
		self.steering_angle = self.compute_new_steering_angle(speed=self.speed, steering_angle=self.steering_angle, compensation_angle=compensation_angle, car_point=self.car_point, car_angle=self.car_angle, next_waypoint=self.car_waypoint)
		# update perceived car point
		self.car_point, self.car_angle = self.compute_new_car_point_and_angle(car_point=self.car_point, car_angle=self.car_angle, steering_angle=self.steering_angle, speed=self.speed)
		# update real car point
		self.noisy_point, _ = self.compute_new_car_point_and_angle(car_point=self.noisy_point, car_angle=self.car_angle, steering_angle=self.steering_angle, speed=self.speed, add_noise=True)
		# update position and direction
		car_position, car_waypoint = self.get_car_position_and_next_waypoint(self.car_point)
		noisy_position, noisy_waypoint = self.get_car_position_and_next_waypoint(self.noisy_point)
		# compute real reward
		noisy_reward = self.get_reward(car_speed=self.speed, car_point=self.noisy_point, car_progress=self.noisy_progress, car_position=noisy_position)
		if noisy_position > self.noisy_progress: # is moving toward next position
			self.noisy_progress = noisy_position # progress update
			self.noisy_waypoint = noisy_waypoint
		# compute perceived reward
		car_reward = self.get_reward(car_speed=self.speed, car_point=self.car_point, car_progress=self.car_progress, car_position=car_position)
		if car_position > self.car_progress: # is moving toward next position
			self.car_progress = car_position # progress update
			self.car_waypoint = car_waypoint
		# compute state (after updating progress)
		state = self.get_state(self.car_point, self.car_waypoint)
		# update last action/state/reward
		self.last_state = state
		self.last_reward = car_reward
		# update cumulative reward
		self.cumulative_reward += noisy_reward
		self.avg_speed_per_steps += self.speed
		# update step
		self.step += 1
		terminal = self.is_terminal_position(self.car_waypoint) or self.is_terminal_position(self.noisy_waypoint) or self.step >= self.max_step
		if terminal: # populate statistics
			stats = {
				"avg_speed": self.avg_speed_per_steps/self.step,
				"reward": self.cumulative_reward,
				"step": self.step,
				"completed": 1 if self.is_terminal_position(self.car_waypoint) else 0
			}
			self.episodes.append(stats)
			if len(self.episodes) > flags.match_count_for_evaluation:
				self.episodes.popleft()
		return state, noisy_reward, terminal
		
	def get_concatenation(self):
		return [self.steering_angle, self.speed, self.last_reward]
		
	def get_reward(self, car_speed, car_point, car_progress, car_position):
		if car_position > car_progress: # is moving toward next position
			distance = euclidean_distance(car_point, self.get_point_from_position(car_position))
			return car_speed*(1 - np.clip(distance/self.max_distance_to_path,0,1)) # always in [0,1] # smaller distances to path give higher rewards
		return -0.1 # is NOT moving toward next position
		
	def get_state(self, car_point, next_waypoint):
		if next_waypoint >= self.path_length:
			next_waypoint = self.path_length-1
		xs, ys = self.path
		next_x, next_y = xs[next_waypoint], ys[next_waypoint]
		car_x, car_y = car_point
		shape = self.get_state_shape()
		state = np.zeros(shape)
		state[0][0] = [car_x, car_y]
		state[0][1] = [next_x, next_y]
		i = 0
		for i in range(4):
			state[0][2+i] = [self.U1[i],self.V1[i]]
		for i in range(4):
			state[0][6+i] = [self.U2[i],self.V2[i]]
		return state
		
	def get_screen(self):
		return self.path
		
	def get_frame_info(self, network, observation, value, action, reward, cross_entropy):
		state_info = "reward={}, speed={}, steering_angle={}, agent={}, value={}, cross_entropy={}\n".format(reward, self.speed, self.steering_angle, network.agent_id, value, cross_entropy)
		action_info = "action={}\n".format(action)
		frame_info = { "log": state_info + action_info }
		if flags.save_episode_screen:
			# First set up the figure, the axis, and the plot element we want to animate
			fig, ax = plt.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10))
			line, = ax.plot(self.path[0], self.path[1], lw=2)
			# ax.plot(0,0,"ro")
			ax.plot(self.noisy_point[0],self.noisy_point[1], marker='o', color='r', label='Real')
			ax.plot(self.car_point[0],self.car_point[1], marker='o', color='b', label='Perceived')
			ax.legend()
			fig.suptitle('Speed: {} \n Angle: {}'.format(self.speed,convert_radiant_to_degree(self.steering_angle)))
			fig.canvas.draw()
			# Now we can save it to a numpy array.
			data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
			data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			frame_info["screen"] = { "value": data, "type": 'RGB' }
			# Release memory
			plt.close(fig)
			# plt.close(ax)
			# gc.collect()
		return frame_info
		
	def get_statistics(self):
		result = {}
		result["avg_reward"] = 0
		result["avg_step"] = 0
		result["avg_speed"] = 0
		result["avg_completed"] = 0
		count = len(self.episodes)
		if count>0:
			for e in self.episodes:
				result["avg_reward"] += e["reward"]
				result["avg_step"] += e["step"]
				result["avg_speed"] += e["avg_speed"]
				result["avg_completed"] += e["completed"]
			result["avg_reward"] /= count
			result["avg_step"] /= count
			result["avg_speed"] /= count
			result["avg_completed"] /= count
		return result
		
def rot(x,y,theta):
	return (x*np.cos(theta)-y*np.sin(theta), x*np.sin(theta)+y*np.cos(theta))

def shift_and_rotate(xv,yv,dx,dy,theta):
	return rot(xv+dx,yv+dy,theta)

def rotate_and_shift(xv,yv,dx,dy,theta):
	(x,y) = rot(xv,yv,theta)
	return (x+dx,y+dy)

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
	return points[0] + points[1]*p + points[2]*p**2 + points[3]*p**3

def derivative(p, points):
	return points[1] + 2*points[2]*p + 3*points[3]*p**2

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

def convert_degree_to_radiant(degree):
	return (degree/180)*np.pi
	
def convert_radiant_to_degree(radiant):
	return radiant*180/np.pi
	
def euclidean_distance(a,b):
	return math.sqrt(sum([(j-k)**2 for (j,k) in zip(a,b)]))