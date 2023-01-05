import numpy as np
import math
from math import *
import tensorflow as tf
import time
from Environment.Model.J import J
from Environment.Model.Vc import Vc
from Environment.Model.WG import WG
from Environment.Model.Rudder import Rudder
from Environment.data_viewer import data_viewer
from Environment.DDPG_test_data_process import data_storage, data_elimation
env_test1 = np.load("C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\test_environments\\env_test1.npy")
env_test2 = np.load("C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\test_environments\\env_test2.npy")
env_test3 = np.load("C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\test_environments\\env_test3.npy")
env_test4 = np.load("C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\test_environments\\env_test4.npy")
env_test5 = np.load("C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\test_environments\\env_test5.npy")

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



    def reset(self, num):
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
        # target relationship
        self.target_position = np.array([100, 100])
        self.distance_target = math.hypot(self.state_0.item(0) - self.target_position[0],
                                          self.state_0.item(1) - self.target_position[1]) / 100
        self.course_error_target = self.state_0.item(3) - self.desired_course(self.target_position[0], self.target_position[1],
                                                                              self.state_0.item(0), self.state_0.item(1))
        # obstacle relationship
        self.obstacle = env_test5[num]

        self.distance_obstacle = np.zeros((1, 5))[0]
        for i in range(5):
            self.distance_obstacle[i] = (math.hypot(self.state_0.item(0) - self.obstacle[i][0],
                                                    self.state_0.item(1) - self.obstacle[i][1])-self.obstacle[i][2]) / 100

        self.course_error_obstacle = np.zeros((1, 5))[0]
        for i in range(5):
            self.course_error_obstacle[i] = self.state_0.item(3) - self.desired_course(self.obstacle[i][0],self.obstacle[i][1],
                                                                                       self.state_0.item(0),self.state_0.item(1))
        # find the nearest obstacle
        i_obstacle = list(self.distance_obstacle).index(min(self.distance_obstacle))

        return np.array([self.distance_target, self.course_error_target, self.distance_obstacle[i_obstacle], self.course_error_obstacle[i_obstacle], 0])

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
        data_storage(self.x1, self.y1, self.phi1, self.t,  u1 = self.u1, rudder_angle = self.Rudder_angle)  # store data in local files
        #print(self.state_0.item(0))
        # target relationship
        self.distance_target = math.hypot(self.state_0.item(0) - self.target_position[0],
                                          self.state_0.item(1) - self.target_position[1]) / 100
        self.course_error_target = self.state_0.item(3) - self.desired_course(self.target_position[0],
                                                                              self.target_position[1],
                                                                              self.state_0.item(0),
                                                                              self.state_0.item(1))
        # obstacle relationship
        for i in range(5):
            self.distance_obstacle[i] = (math.hypot(self.state_0.item(0) - self.obstacle[i][0],
                                                    self.state_0.item(1) - self.obstacle[i][1]) - self.obstacle[i][
                                             2]) / 100
        for i in range(5):
            self.course_error_obstacle[i] = self.state_0.item(3) - self.desired_course(self.obstacle[i][0],
                                                                                       self.obstacle[i][1],
                                                                                       self.state_0.item(0),
                                                                                       self.state_0.item(1))
        #print(self.distance_obstacle)
        # find the nearest obstacle
        i_obstacle = list(self.distance_obstacle).index(min(self.distance_obstacle))
        observation = np.array([self.distance_target, self.course_error_target, max(self.distance_obstacle[i_obstacle]-0.09, 0.01), self.course_error_obstacle[i_obstacle], rudder_angle])
        return observation, self.distance_obstacle[i_obstacle]

    def step(self, action, observation):
        rudder_control = observation[-1] + action
        s_, dis = self.obser(rudder_control)
        reach = 0

        if self.t >= 300:
            done = True
        elif s_[0] < 0.02 + 0.09:
            done = True
            reach = 1
        elif dis <= 0:
            done = True
        else:
            done = False
        return s_, done, reach

    def render(self, num):

        data_viewer(self.x1, self.y1, u1=self.u1, phit=self.phi1, rudder_angle=self.Rudder_angle, t=self._t, xlim_left=-10, xlim_right=120, ylim_left=-10, ylim_right=120,
                        goal_x=100, goal_y=100, num=num)



###############################  DDPG  ####################################
class Actor(object):
    def __init__(self, sess, action_dim, action_bound):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

        self.new_saver = tf.train.Saver()

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('l1'):
                l1 = tf.layers.dense(net, 100, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
            with tf.variable_scope('l2'):
                l2 = tf.layers.dense(l1, 50, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(l2, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a


env = Waveglider()
state_dim = env.n_features
action_dim = 1
action_bound = pi/180
success = 0

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()
#new_saver = tf.train.import_meta_graph('/home/wp/waveglider_RL/Environment/data/Mymodel-100000.meta')
#graph = tf.get_default_graph()
actor = Actor(sess, action_dim, action_bound)
new_saver = actor.new_saver
new_saver.restore(sess, tf.train.latest_checkpoint('C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\Strategy10'))

for num in range(100):

    s = env.reset(num)
    s = s[np.newaxis, :]
    while True:
        #env.render(num)
        a = sess.run(actor.a, feed_dict={S: s})[0]
        s_, done, reach = env.step(float(a), s[0])
        if reach == 1:
            success += 1

        s = s_
        s = s[np.newaxis, :]

        if done:
            break

print(success, 'Amazing! Eve give you a kiss!!!')