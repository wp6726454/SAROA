import numpy as np
import time
import math
from math import *
from Environment.Model.J import J
from Environment.Model.Vc import Vc
from Environment.Model.WG import WG
from Environment.Model.Rudder import Rudder
from Environment.data_viewer import data_viewer
from Environment.data_process import data_storage, data_elimation

class Waveglider(object):
    # initialization of data storage lists
    def __init__(self):
        self.n_features = 5
        self._t = []
        self.time_step = 0.1
        # sea state
        self.H = 0.3
        self.omega = 1
        self.c_dir = 0
        self.c_speed = 0
        self.state_0 = np.zeros((8, 1))


        # float
        self.x1 = []
        self.y1 = []
        self.z1 = []
        self.phi1 = []
        self.u1 = []
        self.v1 = []
        self.w1 = []
        self.r1 = []

        # forces
        self.Thrust = []
        self.Rudder_angle = []
        self.Frudder_x = []
        self.Frudder_y = []
        self.Frudder_n = []
        #target position
        self.target_position = np.array([100, 100])
        self.obstacle = np.zeros((1, 2), dtype=int)[0]
        self.obst_R = 0

    def reset(self):
        time.sleep(0.1)
        data_elimation()  # Turn on when previous data needs to be cleared
        self.t = 0
        self._t.clear()
        # float
        self.x1.clear()
        self.y1.clear()
        self.z1.clear()
        self.phi1.clear()
        self.u1.clear()
        self.v1.clear()
        self.w1.clear()
        self.r1.clear()

        # forces
        self.Thrust.clear()

        self.Rudder_angle.clear()
        self.Frudder_x.clear()
        self.Frudder_y.clear()
        self.Frudder_n.clear()
        # initial state
        self.state_0 = np.array([[0], [0], [0], [0],  # eta1
                            [0], [0], [0], [0]],  float)  # V1
        # random obstacle
        self.obstacle = np.random.randint(15, 85, 2)
        self.obst_R = np.random.randint(8, 15)

        self.distance_target = math.hypot(self.state_0.item(0) - self.target_position[0],self.state_0.item(1) - self.target_position[1]) / 100
        self.distance_obstacle = (math.hypot(self.state_0.item(0) - self.obstacle[0],self.state_0.item(1) - self.obstacle[1])-self.obst_R) / 100

        self.course_error_target = self.state_0.item(3) - self.desired_course(self.target_position[0],self.target_position[1],self.state_0.item(0),self.state_0.item(1))
        self.course_error_obstacle = self.state_0.item(3) - self.desired_course(self.obstacle[0],self.obstacle[1],self.state_0.item(0),self.state_0.item(1))

        return np.array([self.distance_target, self.course_error_target, self.distance_obstacle, self.course_error_obstacle, 0])

    def desired_course(self,setpoint_x,setpoint_y,realposition_x,realposition_y):

        '''calculate the desired course based on the real-time location and set point'''

        if setpoint_x == realposition_x and setpoint_y > realposition_y:
            phid = pi/2
        elif setpoint_x == realposition_x and setpoint_y < realposition_y:
            phid = -pi/2
        elif setpoint_x > realposition_x and setpoint_y >= realposition_y:
            phid = atan((setpoint_y-realposition_y)/(setpoint_x-realposition_x))
        elif setpoint_x < realposition_x and setpoint_y >= realposition_y:
            phid = atan((setpoint_y-realposition_y)/(setpoint_x-realposition_x)) + pi
        elif setpoint_x < realposition_x and setpoint_y < realposition_y:
            phid = atan((setpoint_y-realposition_y)/(setpoint_x-realposition_x)) - pi
        else:
            phid = atan((setpoint_y-realposition_y)/(setpoint_x-realposition_x))

        return (phid)

    def f(self, state, angle):
        #  float's position and attitude vector
        eta1 = state[0:4]
        #eta1[2] = self.H / 2 * sin(self.omega * t)
        WF = np.array([[20], [0], [0], [0]])
        #  float's velocity vector
        V1 = state[4:8]

        #  float's relative velocity vector
        V1_r = V1 - Vc(self.c_dir, self.c_speed, eta1)
        wg = WG(eta1, eta1, V1, V1, self.c_dir, self.c_speed)
        rudder = Rudder(eta1, V1, self.c_dir, self.c_speed)
        # float's kinematic equations
        eta1_dot = np.dot(J(eta1), V1)

        Minv_1 = np.linalg.inv(wg.MRB_1() + wg.MA_1())

        MV1_dot = - np.dot(wg.CRB_1(), V1) - np.dot(wg.CA_1(), V1_r) - np.dot(wg.D_1(), V1_r) - wg.d_1() + rudder.force(angle) + WF

        V1_dot = np.dot(Minv_1, MV1_dot)

        return np.vstack((eta1_dot, V1_dot))

    def change_angle(self, degree):
        if degree > pi:
            output = degree - 2*pi
        elif degree < -pi:
            output = degree + 2*pi
        else:
            output = degree
        return output

    def obser(self, rudder_angle):

        for _ in range(0, 10, 1):
            # Runge-Kutta
            k1 = self.f(self.state_0, rudder_angle)* self.time_step
            k2 = self.f(self.state_0 + 0.5 * k1, rudder_angle)* self.time_step
            k3 = self.f(self.state_0 + 0.5 * k2, rudder_angle, )* self.time_step
            k4 = self.f(self.state_0 + k3, rudder_angle)* self.time_step
            self.state_0 += (1/6)*(k1 + 2 * k2 + 2 * k3 + k4)
            self.state_0[3] = self.change_angle(self.state_0.item(3))
            self.t += 0.1

        self._t.append(self.t)
        self.x1.append(self.state_0.item(0))
        self.y1.append(self.state_0.item(1))
        self.z1.append(self.state_0.item(2))
        self.phi1.append(self.state_0.item(3))
        self.u1.append(self.state_0.item(4))
        self.v1.append(self.state_0.item(5))
        self.w1.append(self.state_0.item(6))
        self.r1.append(self.state_0.item(7))
        self.Rudder_angle.append(rudder_angle)
        data_storage(self.x1, self.y1, self.phi1, self.t, u1 = self.u1, rudder_angle = self.Rudder_angle)  # store data in local files

        self.distance_target = math.hypot(self.state_0.item(0) - self.target_position[0], self.state_0.item(1) - self.target_position[1]) / 100
        self.distance_obstacle = (math.hypot(self.state_0.item(0) - self.obstacle[0], self.state_0.item(1) - self.obstacle[1]) - self.obst_R) / 100

        self.course_error_target = self.state_0.item(3) - self.desired_course(self.target_position[0], self.target_position[1], self.state_0.item(0), self.state_0.item(1))
        self.course_error_obstacle = self.state_0.item(3) - self.desired_course(self.obstacle[0], self.obstacle[1], self.state_0.item(0), self.state_0.item(1))

        observation = np.array([self.distance_target, self.course_error_target, self.distance_obstacle, self.course_error_obstacle, rudder_angle])
        return observation

    def step(self, action, observation):
        rudder_control = observation[-1] + action
        s_ = self.obser(rudder_control)

        # reward function
        # real_position = s_[:2]
        # pre_real_position = observation[:2]
        #
        # distance_1 = self.target_position - real_position
        # distance = math.hypot(distance_1[0], distance_1[1])
        #
        # pre_distance_1 = self.target_position - pre_real_position
        # pre_distance = math.hypot(pre_distance_1[0], pre_distance_1[1])
        reach = 0

        if self.t >= 250:
            reward = -5
            done = True
        elif s_[0] < 0.02:
            reach = 1
            reward =100
            done = True
        elif s_[2] <= 0:
            reward = -1500
            done = True
        else:
            reward = -1200*(s_[0]-observation[0])-12*(s_[1]-observation[1])+100*(s_[2]-observation[2])+4*(s_[3]-observation[3])-5
            done = False

        return s_, reward, done, reach

    def render(self):

        data_viewer(self.x1, self.y1, u1=self.u1, phit=self.phi1, rudder_angle=self.Rudder_angle, t=self._t, xlim_left=-10, xlim_right=120, ylim_left=-10, ylim_right=120,
                        goal_x=100, goal_y=100, obs_x=self.obstacle[0], obs_y=self.obstacle[1], obs_R=self.obst_R)

