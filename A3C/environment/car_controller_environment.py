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
		return (1,5,2)

	def get_action_shape(self):
		return (2,) # steering angle, continuous control without softmax
	
	def __init__(self, thread_index):
		Environment.__init__(self)
		self.thread_index = thread_index
		self.max_step = 100
		self.spline_number = 2
		self.control_points_per_spline = 50
		self.seconds_per_step = 0.1 # a step every n seconds
		# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
		self.min_speed = 0.1 # m/s
		self.max_speed = 1.4 # m/s
		self.max_speed_noise = 0.25 # m/s
		# the fastest car has max_acceleration 9.25 m/s (https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration)
		# the slowest car has max_acceleration 0.7 m/s (http://automdb.com/max_acceleration)
		self.max_acceleration = 0.7 # m/s
		self.horizon_distance = 2 # meters
		self.max_distance_to_path = 0.1 # meters
		self.max_steering_degree = 30
		self.max_steering_angle = convert_degree_to_radiant(self.max_steering_degree)
		# evaluator stuff
		self.episodes = deque()
	
	def reset(self):
		self.path = self.build_random_path()
		self.car_point = (0,0) # car point and orientation are always expressed with respect to the initial point and orientation of the road fragment
		self.car_progress, self.car_goal = self.get_car_position_and_goal(self.car_point)
		self.speed = self.min_speed + (self.max_speed-self.min_speed)*np.random.random() # random initial speed in [0,max_speed]
		self.car_angle = self.get_angle_from_position(self.car_progress)
		self.steering_angle = 0
		self.cumulative_reward = 0
		self.step = 0
		self.last_reward = 0
		self.last_state = self.get_state(car_point=self.car_point, car_angle=self.car_angle, car_progress=self.car_progress, car_goal=self.car_goal)
		self.avg_speed_per_steps = 0
		
	def get_point_from_position(self, position):
		spline = int(np.ceil(position)-1)
		if spline <= 0: # first spline 
			return (poly(position,self.U[0]), poly(position,self.V[0]))
		# second spline
		return rotate_and_shift(poly(position-spline,self.U[spline]), poly(position-spline,self.V[spline]), self.middle_point[spline-1][0], self.middle_point[spline-1][1], self.theta[spline-1])
		
	def get_angle_from_position(self, position):
		spline = int(np.ceil(position)-1)
		if spline <= 0: # first spline 
			return angle(position, self.U[0], self.V[0])
		# second spline
		return angle(position-spline, self.U[spline], self.V[spline])+self.theta[spline-1]
		
	def build_random_path(self):
		# setup environment
		self.U = []
		self.V = []
		self.theta = []
		self.middle_point = []
		for i in range(self.spline_number):
			U, V = generate_random_polynomial()
			self.U.append(U)
			self.V.append(V)
			self.theta.append(angle(1, U, V))
			self.middle_point.append(self.get_point_from_position(i+1))
		# we generate all points for both polynomials, then we shall draw only a portion of them
		self.positions = np.linspace(start=0, stop=self.spline_number, num=self.spline_number*self.control_points_per_spline) # first spline is in [0,1] while the second one is in [1,2]
		xy = [self.get_point_from_position(pos) for pos in self.positions]
		return list(zip(*xy))

	def is_terminal_position(self, position):
		return position >= self.spline_number*0.9

	def get_car_position_and_goal(self, car_point):
		car_x, car_y = car_point
		# Find the closest spline point
		car_closest_position = optimize.minimize_scalar(lambda pos: euclidean_distance(car_point, self.get_point_from_position(pos)), method='bounded', bounds=(0,self.spline_number))
		car_position = car_closest_position.x
		# Find closest control point on horizon
		closest_goal = optimize.minimize_scalar(lambda pos: np.absolute(euclidean_distance(car_point, self.get_point_from_position(pos))-self.horizon_distance), method='bounded', bounds=(car_position,self.spline_number))
		goal = closest_goal.x
		return car_position, goal

	def move_car(self, car_point, car_angle, steering_angle, speed, add_noise=False):
		# get new car angle
		new_car_angle = car_angle + steering_angle
		# change point
		car_x, car_y = car_point
		speed_noise = (2*np.random.random()-1)*self.max_speed_noise if add_noise else 0
		space = (speed+speed_noise)*self.seconds_per_step
		dir_x, dir_y = get_heading_vector(angle=new_car_angle, space=space)
		return (car_x+dir_x, car_y+dir_y), new_car_angle

	def compute_new_steering_angle(self, action, speed, steering_angle, car_point, car_angle, car_goal): # action is in [-1,1]
		# get new steering angle
		return action*self.max_steering_angle # in [-max_steering_angle, max_steering_angle]
		
	def compute_new_speed(self, action, speed): # action is in [-1,1]
		speed += action*self.max_acceleration*self.seconds_per_step
		return np.clip(speed, self.min_speed, self.max_speed)

	def process(self, action_vector):
		# compute new speed
		self.speed = self.compute_new_speed(action=action_vector[1], speed=self.speed)
		# compute new steering_angle
		self.steering_angle = self.compute_new_steering_angle(action=action_vector[0], speed=self.speed, steering_angle=self.steering_angle, car_point=self.car_point, car_angle=self.car_angle, car_goal=self.car_goal)
		# move car
		self.car_point, self.car_angle = self.move_car(car_point=self.car_point, car_angle=self.car_angle, steering_angle=self.steering_angle, speed=self.speed, add_noise=True)
		# update position and direction
		car_position, car_goal = self.get_car_position_and_goal(self.car_point)
		# compute perceived reward
		reward = self.get_reward(car_speed=self.speed, car_point=self.car_point, car_progress=self.car_progress, car_position=car_position)
		if car_position > self.car_progress: # is moving toward next position
			self.car_progress = car_position # progress update
			self.car_goal = car_goal
		# compute state (after updating progress)
		state = self.get_state(car_point=self.car_point, car_angle=self.car_angle, car_progress=self.car_progress, car_goal=self.car_goal)
		# update last action/state/reward
		self.last_state = state
		self.last_reward = reward
		# update cumulative reward
		self.cumulative_reward += reward
		self.avg_speed_per_steps += self.speed
		# update step
		self.step += 1
		terminal = self.is_terminal_position(self.car_goal) or self.step >= self.max_step
		if terminal: # populate statistics
			stats = {
				"avg_speed": self.avg_speed_per_steps/self.step,
				"reward": self.cumulative_reward,
				"step": self.step,
				"completed": 1 if self.is_terminal_position(self.car_goal) else 0
			}
			self.episodes.append(stats)
			if len(self.episodes) > flags.match_count_for_evaluation:
				self.episodes.popleft()
		return state, reward, terminal
	
	def get_concatenation_size(self):
		return 3
		
	def get_concatenation(self):
		return [self.steering_angle, self.speed, self.last_reward]
		
	def get_reward(self, car_speed, car_point, car_progress, car_position):
		if car_position > car_progress: # is moving toward next position
			distance = euclidean_distance(car_point, self.get_point_from_position(car_position))
			return car_speed*self.seconds_per_step*(1 - np.clip(distance/self.max_distance_to_path,0,1)) # always in [0,1] # smaller distances to path give higher rewards
		return -0.1 # is NOT moving toward next position
		
	def get_state(self, car_point, car_angle, car_progress, car_goal):
		shape = self.get_state_shape()
		state = np.zeros(shape)
		car_x, car_y = car_point
		control_distance = (car_goal - car_progress)/shape[1]
		for i in range(shape[1]):
			# get points coordinates relative to car (car is zero)
			xa, ya = self.get_point_from_position(car_progress + (i+1)*control_distance)
			state[0][i] = shift_and_rotate(xa, ya, -car_x, -car_y, -car_angle) # point relative to car position
		return state
		
	def get_screen(self): # RGB array
		# First set up the figure, the axis, and the plot element we want to animate
		fig, ax = plt.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10))
		ax.plot(self.path[0], self.path[1], lw=2, label='Path')
		# ax.plot(0,0,"ro")
		car_x, car_y = self.car_point
		ax.plot(car_x, car_y, marker='o', color='g', label='Car')
		dir_x, dir_y = get_heading_vector(angle=self.car_angle)
		ax.plot([car_x, car_x+dir_x],[car_y, car_y+dir_y], color='g', label='Heading Vector')
		waypoint_x, waypoint_y = self.get_point_from_position(self.car_goal)
		ax.plot(waypoint_x, waypoint_y, marker='o', color='r', label='Waypoint')
		# progress_x, progress_y = self.get_point_from_position(self.car_progress)
		# ax.plot(progress_x, progress_y, marker='o', color='y', label='Progress')
		ax.legend()
		fig.suptitle('Speed: {0:.2f} m/s \n Angle: {1:.2f} deg \n Step: {2}'.format(self.speed,convert_radiant_to_degree(self.steering_angle), self.step))
		fig.canvas.draw()
		# Now we can save it to a numpy array.
		data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		# Release memory
		plt.close(fig)
		# plt.close(ax)
		# gc.collect()
		return data # RGB array
		
	def get_frame_info(self, network, observation, value, action, reward, policy):
		state_info = "reward={}, speed={}, steering_angle={}, agent={}, value={}, policy={}\n".format(reward, self.speed, self.steering_angle, network.agent_id, value, policy)
		state_info += "state={}\n".format(self.last_state)
		action_info = "action={}\n".format(action)
		frame_info = { "log": state_info + action_info }
		if flags.save_episode_screen:
			frame_info["screen"] = { "value": observation, "type": 'RGB' }
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
	
def get_heading_vector(angle, space=1):
	return (space*np.cos(angle), space*np.sin(angle))
	
def euclidean_distance(a,b):
	return math.sqrt(sum([(j-k)**2 for (j,k) in zip(a,b)]))