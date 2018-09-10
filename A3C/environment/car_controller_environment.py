# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

import numpy as np
from scipy import optimize
from collections import deque
from environment.environment import Environment

import options
flags = options.get() # get command line args
	
class CarControllerEnvironment(Environment):

	def get_state_shape(self):
		# There are 2 types of objects (obstacles and lanes), each object has 3 numbers (x, y and size)
		if self.max_obstacle_count > 0:
			return (2,max(self.control_points_per_step,self.max_obstacle_count),3)
		return (1,self.control_points_per_step,2) # no need for size because there are only lanes

	def get_action_shape(self):
		return (2,) # steering angle, continuous control without softmax
	
	def __init__(self, thread_index):
		Environment.__init__(self)
		self.thread_index = thread_index
		self.max_step = 100
		self.control_points_per_step = 5
		self.mean_seconds_per_step = 0.1 # in average, a step every n seconds
		self.horizon_distance = 1 # meters
		self.max_distance_to_path = 0.1 # meters
		# obstacles related stuff
		self.max_obstacle_count = 3
		self.min_obstacle_radius = 0.15 # meters
		self.max_obstacle_radius = 0.45 # meters
		# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
		self.min_speed = 0.1 # m/s
		self.max_speed = 1.4 # m/s
		self.speed_lower_limit = 0.7 # m/s # used together with max_speed to get the random speed upper limit
		self.max_speed_noise = 0.25 # m/s
		# the fastest car has max_acceleration 9.25 m/s (https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration)
		# the slowest car has max_acceleration 0.7 m/s (http://automdb.com/max_acceleration)
		self.max_acceleration = 0.7 # m/s
		self.max_steering_degree = 30
		self.max_steering_noise_degree = 2
		self.max_steering_angle = convert_degree_to_radiant(self.max_steering_degree)
		self.max_steering_noise_angle = convert_degree_to_radiant(self.max_steering_noise_degree)
		# splines related stuff
		self.spline_number = 2
		self.control_points_per_spline = 50
		# evaluator stuff
		self.episodes = deque()
		# shapes
		self.state_shape = self.get_state_shape()
		self.action_shape = self.get_action_shape()
	
	def reset(self):
		self.step = 0
		self.seconds_per_step = self.get_step_seconds()
		self.path = self.build_random_path()
		# car position
		self.car_point = (0,0) # car point and orientation are always expressed with respect to the initial point and orientation of the road fragment
		self.car_progress, self.car_goal = self.get_position_and_goal(point=self.car_point)
		self.car_angle = self.get_angle_from_position(self.car_progress)
		# speed limit
		self.speed_upper_limit = self.speed_lower_limit + (self.max_speed-self.speed_lower_limit)*np.random.random() # in [speed_lower_limit,max_speed]
		# steering angle & speed
		self.speed = self.min_speed + (self.max_speed-self.min_speed)*np.random.random() # in [min_speed,max_speed]
		self.steering_angle = 0
		# get obstacles
		self.obstacles = self.get_new_obstacles()
		# init concat variables
		self.last_reward = 0
		self.last_state = self.get_state(car_point=self.car_point, car_angle=self.car_angle, car_progress=self.car_progress, car_goal=self.car_goal, obstacles=self.obstacles)
		# init log variables
		self.cumulative_reward = 0
		self.avg_speed_per_steps = 0
			
	def get_new_obstacles(self):
		if self.max_obstacle_count <= 0:
			return []
		obstacles = []
		presence_mask = np.random.randint(2, size=self.max_obstacle_count)
		for i in range(self.max_obstacle_count):
			if presence_mask[i] == 1: # obstacle is present
				point = self.get_point_from_position(self.spline_number*np.random.random())
				radius = self.min_obstacle_radius + (self.max_obstacle_radius-self.min_obstacle_radius)*np.random.random() # in [min_obstacle_radius,max_obstacle_radius]
				obstacles.append((point,radius))
		return obstacles
		
	def get_closest_obstacle(self, point, obstacles):
		if len(obstacles) == 0:
			return None
		obstacle_distances_from_point = map(lambda obstacle: (obstacle, euclidean_distance(obstacle[0], point)-obstacle[1]), obstacles)
		return min(obstacle_distances_from_point, key=lambda tup: tup[1])[0]
		
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

	def get_position_and_goal(self, point):
		# Find the closest spline point
		car_closest_position = optimize.minimize_scalar(lambda pos: euclidean_distance(point, self.get_point_from_position(pos)), method='bounded', bounds=(0,self.spline_number))
		car_position = car_closest_position.x
		# Find closest control point on horizon
		closest_goal = optimize.minimize_scalar(lambda pos: np.absolute(euclidean_distance(point, self.get_point_from_position(pos))-self.horizon_distance), method='bounded', bounds=(car_position,self.spline_number))
		goal = closest_goal.x
		return car_position, goal

	def move(self, point, angle, steering_angle, speed, add_noise=False):
		# add noise
		if add_noise:
			steering_angle += (2*np.random.random()-1)*self.max_steering_noise_angle
			steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle) # |steering_angle| <= max_steering_angle, ALWAYS
			speed += (2*np.random.random()-1)*self.max_speed_noise
		# get new angle
		new_angle = angle+steering_angle
		# move point
		x, y = point
		dir_x, dir_y = get_heading_vector(angle=new_angle, space=speed*self.seconds_per_step)
		return (x+dir_x, y+dir_y), new_angle

	def get_steering_angle_from_action(self, action): # action is in [-1,1]
		return action*self.max_steering_angle # in [-max_steering_angle, max_steering_angle]
		
	def get_acceleration_from_action(self, action): # action is in [-1,1]
		return action*self.max_acceleration # in [-max_acceleration, max_acceleration]
		
	def accelerate(self, speed, acceleration):
		return np.clip(speed + acceleration*self.seconds_per_step, self.min_speed, self.max_speed)
		
	def get_step_seconds(self):
		return np.random.exponential(scale=self.mean_seconds_per_step)

	def process(self, action_vector):
		# first of all, get the seconds passed from last step
		self.seconds_per_step = self.get_step_seconds()
		# compute new steering angle
		self.steering_angle = self.get_steering_angle_from_action(action=action_vector[0])
		# compute new acceleration
		self.acceleration = self.get_acceleration_from_action(action=action_vector[1])
		# compute new speed
		self.speed = self.accelerate(speed=self.speed, acceleration=self.acceleration)
		# move car
		self.car_point, self.car_angle = self.move(point=self.car_point, angle=self.car_angle, steering_angle=self.steering_angle, speed=self.speed, add_noise=True)
		# update position and direction
		car_position, car_goal = self.get_position_and_goal(point=self.car_point)
		# compute perceived reward
		reward, dead = self.get_reward(car_speed=self.speed, car_point=self.car_point, car_progress=self.car_progress, car_position=car_position, obstacles=self.obstacles)
		if car_position > self.car_progress: # is moving toward next position
			self.car_progress = car_position # progress update
			self.car_goal = car_goal
		# compute new state (after updating progress)
		state = self.get_state(car_point=self.car_point, car_angle=self.car_angle, car_progress=self.car_progress, car_goal=self.car_goal, obstacles=self.obstacles)
		# update last action/state/reward
		self.last_state = state
		self.last_reward = reward
		# update cumulative reward
		self.cumulative_reward += reward
		self.avg_speed_per_steps += self.speed
		# update step
		self.step += 1
		terminal = dead or self.is_terminal_position(self.car_goal) or self.step >= self.max_step
		if terminal: # populate statistics
			stats = {
				"avg_speed": self.avg_speed_per_steps/self.step,
				"reward": self.cumulative_reward,
				"step": self.step,
				"completed": 1 if self.is_terminal_position(self.car_goal) else 0
			}
			if self.max_obstacle_count > 0:
				stats["hit"] = 1 if dead else 0
			self.episodes.append(stats)
			if len(self.episodes) > flags.match_count_for_evaluation:
				self.episodes.popleft()
		return state, reward, terminal
	
	def get_concatenation_size(self):
		return 4
		
	def get_concatenation(self):
		return [self.steering_angle, self.speed, self.seconds_per_step, self.speed_upper_limit]
		
	def get_reward(self, car_speed, car_point, car_progress, car_position, obstacles):
		max_distance_to_path = self.max_distance_to_path
		car_projection_point = self.get_point_from_position(car_position)
		closest_obstacle = self.get_closest_obstacle(point=car_projection_point, obstacles=obstacles)
		if closest_obstacle is not None:
			obstacle_point, obstacle_radius = closest_obstacle
			if euclidean_distance(obstacle_point, car_point) <= obstacle_radius: # collision
				return (-1, True) # terminate episode
			if euclidean_distance(obstacle_point, car_projection_point) <= obstacle_radius: # could collide obstacle
				max_distance_to_path += obstacle_radius
		if car_position > car_progress: # is moving toward next position
			distance = euclidean_distance(car_point, car_projection_point)
			distance_ratio = np.clip(distance/max_distance_to_path, 0,1) # always in [0,1]
			inverse_distance_ratio = 1 - distance_ratio
			# the more car_speed > self.speed_upper_limit, the bigger the malus
			malus = self.speed_upper_limit*max(0,car_speed/self.speed_upper_limit-1)*self.seconds_per_step
			# smaller distances to path give higher rewards
			bonus = min(car_speed,self.speed_upper_limit)*self.seconds_per_step*inverse_distance_ratio
			return (bonus-malus, False) # do not terminate episode
		# else is NOT moving toward next position
		return (-0.1, False) # do not terminate episode
		
	def get_state(self, car_point, car_angle, car_progress, car_goal, obstacles):
		state = np.zeros(self.state_shape)
		car_x, car_y = car_point
		control_distance = (car_goal - car_progress)/self.control_points_per_step
		# add control points
		for i in range(self.control_points_per_step):
			cp_x, cp_y = self.get_point_from_position(car_progress + (i+1)*control_distance)
			rcp_x, rcp_y = shift_and_rotate(cp_x, cp_y, -car_x, -car_y, -car_angle) # get control point with coordinates relative to car point
			if self.max_obstacle_count > 0:
				state[0][i] = (rcp_x, rcp_y, 0) # no collision with lanes
			else:
				state[0][i] = (rcp_x, rcp_y)
		# add obstacles
		for (j, obstacle) in enumerate(obstacles):
			obstacle_point, obstacle_radius = obstacle
			if euclidean_distance(obstacle_point,car_point) <= self.horizon_distance+obstacle_radius:
				ro_x, ro_y = shift_and_rotate(obstacle_point[0], obstacle_point[1], -car_x, -car_y, -car_angle) # get control point with coordinates relative to car point
				state[1][j] = (ro_x, ro_y, obstacle_radius)
		return state
		
	def get_screen(self): # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(5,5))
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
		# [Obstacles]
		if len(self.obstacles) > 0:
			circles = [Circle(point,radius,color='b') for (point,radius) in self.obstacles]
			patch_collection = PatchCollection(circles, match_original=True)
			ax.add_collection(patch_collection)
		# [Car]
		car_x, car_y = self.car_point
		car_handle = ax.scatter(car_x, car_y, marker='o', color='g', label='Car')
		# [Heading Vector]
		dir_x, dir_y = get_heading_vector(angle=self.car_angle)
		heading_vector_handle, = ax.plot([car_x, car_x+dir_x],[car_y, car_y+dir_y], color='g', alpha=0.5, label='Heading Vector')
		# [Goal]
		waypoint_x, waypoint_y = self.get_point_from_position(self.car_goal)
		goal_handle = ax.scatter(waypoint_x, waypoint_y, marker='o', color='r', label='Horizon')
		# [Path]
		path_handle, = ax.plot(self.path[0], self.path[1], lw=2, alpha=0.5, label='Path')
		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		handles = [car_handle,heading_vector_handle,goal_handle,path_handle]
		if len(self.obstacles) > 0:
			# https://stackoverflow.com/questions/11423369/matplotlib-legend-circle-markers
			handles.append(Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="blue", label='Obstacle'))
		ax.legend(handles=handles)
		# Draw plot
		figure.suptitle('[Speed]{0:.2f} m/s [Angle]{1:.2f} deg \n [Limit]{3:.2f} m/s [Step]{2}'.format(self.speed,convert_radiant_to_degree(self.steering_angle), self.step, self.speed_upper_limit))
		canvas.draw()
		# Save plot into RGB array
		data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
		return data # RGB array
		
	def get_frame_info(self, network, value, action, reward, policy):
		state_info = "reward={}, speed={}, steering_angle={}, agent={}, value={}, policy={}\n".format(reward, self.speed, self.steering_angle, network.agent_id, value, policy)
		state_info += "state={}\n".format(self.last_state)
		action_info = "action={}\n".format(action)
		frame_info = { "log": state_info + action_info }
		if flags.save_episode_screen:
			frame_info["screen"] = { "value": self.get_screen(), "type": 'RGB' }
		return frame_info
		
	def get_statistics(self):
		result = {}
		result["avg_reward"] = 0
		result["avg_step"] = 0
		result["avg_speed"] = 0
		result["avg_completed"] = 0
		if self.max_obstacle_count > 0:
			result["avg_hit"] = 0
		count = len(self.episodes)
		if count>0:
			result["avg_reward"] = sum(e["reward"] for e in self.episodes)/count
			result["avg_step"] = sum(e["step"] for e in self.episodes)/count
			result["avg_speed"] = sum(e["avg_speed"] for e in self.episodes)/count
			result["avg_completed"] = sum(e["completed"] for e in self.episodes)/count
			if self.max_obstacle_count > 0:
				result["avg_hit"] = sum(e["hit"] for e in self.episodes)/count
		return result
		
def rotate(x,y,theta):
	return (x*np.cos(theta)-y*np.sin(theta), x*np.sin(theta)+y*np.cos(theta))

def shift_and_rotate(xv,yv,dx,dy,theta):
	return rotate(xv+dx,yv+dy,theta)

def rotate_and_shift(xv,yv,dx,dy,theta):
	(x,y) = rotate(xv,yv,theta)
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
	return radiant*(180/np.pi)
	
def get_heading_vector(angle, space=1):
	return (space*np.cos(angle), space*np.sin(angle))
	
def euclidean_distance(a,b):
	return np.sqrt(sum((j-k)**2 for (j,k) in zip(a,b)))