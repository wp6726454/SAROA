import numpy as np
import math
from math import *
import tensorflow as tf
import time
from Environment.Model.J import J
from Environment.Model.Vc import Vc
from Environment.Model.WG import WG
from Environment.Model.Rudder import Rudder
from Environment.pf_oa_viewer import data_viewer
from Environment.data_process import data_storage, data_elimation
from Environment.Model.PID import PID
# env_obstacle = np.load("C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\test_environments\\env_test.npy")
env_obstacle = np.array([
 [106, 754, 14],
 [213, 109, 15],
 [540, 548, 17],
 [178, 436, 11],
 [288,  91, 18],
 [600, 400, 19],
 [400, 609, 11],
 [100, 800, 10],
 [800, 100, 10],
 [150, 900, 12],
 [900, 150, 12],
 [800, 300, 17],
 [300, 800, 17],
 [300, 700, 18],
 [700, 200, 18],
 [750, 400, 12],
 [400, 750, 12],
 [300, 300, 12],
 [950, 200, 12],
 [850, 500, 15],
 [550, 900, 15],
 [50, 950, 14],
 [750, 750, 15],
 [980, 100, 11],
 [100, 980, 10],
 [200, 880, 12],
 [790, 180, 12],
 [819, 857,  13],
 [557, 699,  14],
 [927, 715,  11],
 [327, 567,  12.],
 [400, 368, 16.],
 [189, 184, 10.],
 [710, 700, 13.],
 [738, 835, 15.],
 [1000, 800,  17.],
 [358, 178,  18.],
 [535, 264,  14.],
 [500, 451,  15.],
 [301, 453,  16.],
 [690, 419,  19.],
 [59, 53,  12.],
 [941, 908, 14.],
 [150, 500, 18.],
 [74, 258,  18.],
 [550, 163, 16.],
 [343, 597, 16.],
 [283, 223, 14.],
 [400, 1000, 17.],
 [156,  46, 18.]])
pointsway = np.array([[501, 500], [1000, 1000]])

# env_obstacle = np.array([[50, 50, 17],
# [150, 150, 22],
# [250, 250,  19],
# [ 350, 350,  12],
# [ 450, 450,  18],
# [ 550,  550,  24],
# [ 650, 650,   25],
# [ 750, 750,  28],
# [ 850, 850,  20],
# [950, 950, 21]])


#pointsway = np.array([[1, 0], [1000, 1000]])

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
        self.pointsway = pointsway
        self.deta =80
        # obstacel perception range


        # float
        self.x1 = []
        self.y1 = []
        self.z1 = []
        self.phi1 = []
        self.u1 = []
        self.v1 = []
        self.w1 = []
        self.r1 = []
        self.e1 = []
        self.d1 = []
        # forces
        self.Thrust = []
        self.Rudder_angle = []
        self.Frudder_x = []
        self.Frudder_y = []
        self.Frudder_n = []



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
        self.e1.clear()
        self.d1.clear()
        # forces
        self.Thrust.clear()
        self.Rudder_angle.clear()
        self.Frudder_x.clear()
        self.Frudder_y.clear()
        self.Frudder_n.clear()
        # initial state
        self.state_0 = np.array([[500], [500], [0], [1],  # eta1
                            [0], [0], [0], [0]],  float)  # V1
        # target relationship
        self.target_position = np.array([1000, 1000])
        self.distance_target = math.hypot(self.state_0.item(0) - self.target_position[0],
                                          self.state_0.item(1) - self.target_position[1]) / 100
        #LOS position relationship
        self.LOS_position = self.p_f()
        self.distance_LOS = math.hypot(self.state_0.item(0) - self.LOS_position[0],
                                          self.state_0.item(1) - self.LOS_position[1]) / 100
        self.course_LOS = self.desired_course(self.LOS_position[0], self.LOS_position[1],
                                            self.state_0.item(0), self.state_0.item(1))
        self.course_error_LOS = self.state_0.item(3) - self.course_LOS
        # obstacle relationship
        self.obstacle = env_obstacle
        print(self.obstacle)
        #print(self.obstacle)
        self.distance_obstacle = np.zeros((1, 50))[0]
        # for i in range(50):
        #     self.distance_obstacle[i] = (math.hypot(self.state_0.item(0) - self.obstacle[i][0],
        #                                             self.state_0.item(1) - self.obstacle[i][1])-self.obstacle[i][2]) / 100

        # find the nearest obstacle
        i_obstacle = list(self.distance_obstacle).index(min(self.distance_obstacle))
        self.course_obstacle = self.desired_course(self.obstacle[i_obstacle][0], self.obstacle[i_obstacle][1],
                                              self.state_0.item(0), self.state_0.item(1))
        self.course_error_obstacle = self.state_0.item(3) - self.course_obstacle
        return np.array([self.distance_LOS, self.course_error_LOS, self.distance_obstacle[i_obstacle], self.course_error_obstacle, 0]), self.state_0.item(3), self.course_LOS

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

    def p_f(self):
        # find LOS posiiton
        # judge current track line
        self.realposition = np.array([self.state_0.item(1), self.state_0.item(0)])
        #print(self.realposition)
        d = np.zeros(len(self.pointsway))
        for i in range(len(self.pointsway)):
            d[i] = np.linalg.norm(self.pointsway[i] - self.realposition)
        d = list(d)
        b = d.index(min(d))  # find the nearest waypoint
        if b == 0:
            A = self.pointsway[b + 1, 1] - self.pointsway[b, 1]
            B = self.pointsway[b, 0] - self.pointsway[b + 1, 0]
            C = self.pointsway[b + 1, 0] * self.pointsway[b, 1] - self.pointsway[b, 0] * self.pointsway[b + 1, 1]
            # calculate chuizu
            y0 = (B * B * self.realposition[0] - A * B * self.realposition[1] - A * C) / (A * A + B * B)
            x0 = (-A * B * self.realposition[0] + A * A * self.realposition[1] - B * C) / (A * A + B * B)
            # yd = y0 - np.sign(self.pointsway[b, 0] - self.pointsway[b + 1, 0]) * np.sqrt(np.square(self.deta) / np.square(1 + (abs(self.pointsway[b, 1] - self.pointsway[b + 1, 1]) / (self.pointsway[b, 0] - self.pointsway[b + 1, 0]))))
            # xd = x0 - np.sign(self.pointsway[b, 1] - self.pointsway[b + 1, 1]) * (abs((self.pointsway[b, 1] - self.pointsway[b + 1, 1]) / (self.pointsway[b, 0] - self.pointsway[b + 1, 0]))) * (y0 - yd)
            yd = y0 + self.deta * ((self.pointsway[b+1, 0]-self.pointsway[b, 0]) / (np.sqrt(np.square(self.pointsway[b+1, 0]-self.pointsway[b, 0])+np.square(self.pointsway[b+1, 1]-self.pointsway[b, 1]))))
            xd = x0 + self.deta * ((self.pointsway[b+1, 1]-self.pointsway[b, 1]) /(np.sqrt(np.square(self.pointsway[b+1, 0]-self.pointsway[b, 0])+np.square(self.pointsway[b+1, 1]-self.pointsway[b, 1]))))

        elif b == len(self.pointsway) - 1:

            A = self.pointsway[b, 1] - self.pointsway[b - 1, 1]
            B = self.pointsway[b - 1, 0] - self.pointsway[b, 0]
            C = self.pointsway[b, 0] * self.pointsway[b - 1, 1] - self.pointsway[b - 1, 0] * self.pointsway[b, 1]
            # calculate chuizu
            y0 = (B * B * self.realposition[0] - A * B * self.realposition[1] - A * C) / (A * A + B * B)
            x0 = (-A * B * self.realposition[0] + A * A * self.realposition[1] - B * C) / (A * A + B * B)
            # yd = y0 - np.sign(self.pointsway[b - 1, 0] - self.pointsway[b, 0]) * np.sqrt(np.square(self.deta) / np.square(1 + (abs(self.pointsway[b - 1, 1] - self.pointsway[b, 1]) / (self.pointsway[b - 1, 0] - self.pointsway[b, 0]))))
            # xd = x0 - np.sign(self.pointsway[b - 1, 1] - self.pointsway[b, 1]) * (abs((self.pointsway[b - 1, 1] - self.pointsway[b, 1]) / (self.pointsway[b - 1, 0] - self.pointsway[b, 0]))) * (y0 - yd)
            yd = y0 + self.deta * ((self.pointsway[b, 0] - self.pointsway[b-1, 0]) / (np.sqrt(
                np.square(self.pointsway[b, 0] - self.pointsway[b-1, 0]) + np.square(
                    self.pointsway[b, 1] - self.pointsway[b-1, 1]))))
            xd = x0 + self.deta * ((self.pointsway[b, 1] - self.pointsway[b-1, 1]) / (np.sqrt(
                np.square(self.pointsway[b, 0] - self.pointsway[b-1, 0]) + np.square(
                    self.pointsway[b, 1] - self.pointsway[b-1, 1]))))

        else:

            PbPt = self.realposition - self.pointsway[b]
            PbPb_1 = self.pointsway[b - 1] - self.pointsway[b]
            PbPb1 = self.pointsway[b + 1] - self.pointsway[b]

            if self.angle(PbPt, PbPb_1) < self.angle(PbPt, PbPb1):
                A = self.pointsway[b, 1] - self.pointsway[b - 1, 1]
                B = self.pointsway[b - 1, 0] - self.pointsway[b, 0]
                C = self.pointsway[b, 0] * self.pointsway[b - 1, 1] - self.pointsway[b - 1, 0] * self.pointsway[b, 1]
                # calculate chuizu
                y0 = (B * B * self.realposition[0] - A * B * self.realposition[1] - A * C) / (A * A + B * B)
                x0 = (-A * B * self.realposition[0] + A * A * self.realposition[1] - B * C) / (A * A + B * B)
                yd = y0 + self.deta * ((self.pointsway[b, 0] - self.pointsway[b - 1, 0]) / (np.sqrt(
                    np.square(self.pointsway[b, 0] - self.pointsway[b - 1, 0]) + np.square(
                        self.pointsway[b, 1] - self.pointsway[b - 1, 1]))))
                xd = x0 + self.deta * ((self.pointsway[b, 1] - self.pointsway[b - 1, 1]) / (np.sqrt(
                    np.square(self.pointsway[b, 0] - self.pointsway[b - 1, 0]) + np.square(
                        self.pointsway[b, 1] - self.pointsway[b - 1, 1]))))


            else:

                A = self.pointsway[b + 1, 1] - self.pointsway[b, 1]
                B = self.pointsway[b, 0] - self.pointsway[b + 1, 0]
                C = self.pointsway[b + 1, 0] * self.pointsway[b, 1] - self.pointsway[b, 0] * self.pointsway[b + 1, 1]
                # calculate chuizu
                y0 = (B * B * self.realposition[0] - A * B * self.realposition[1] - A * C) / (A * A + B * B)
                x0 = (-A * B * self.realposition[0] + A * A * self.realposition[1] - B * C) / (A * A + B * B)
                yd = y0 + self.deta * ((self.pointsway[b + 1, 0] - self.pointsway[b, 0]) / (np.sqrt(
                    np.square(self.pointsway[b + 1, 0] - self.pointsway[b, 0]) + np.square(
                        self.pointsway[b + 1, 1] - self.pointsway[b, 1]))))
                xd = x0 + self.deta * ((self.pointsway[b + 1, 1] - self.pointsway[b, 1]) / (np.sqrt(
                    np.square(self.pointsway[b + 1, 0] - self.pointsway[b, 0]) + np.square(
                        self.pointsway[b + 1, 1] - self.pointsway[b, 1]))))
        e1 = np.sqrt(np.square(y0 - self.realposition[0]) + np.square(x0 - self.realposition[1]))
        self.e1.append(e1)
        return np.array([xd, yd])

    def angle(self, a, b):
        c = np.dot(a, b)
        d = np.dot(a, a)
        e = np.sqrt(d)
        f = np.dot(b, b)
        g = np.sqrt(f)
        h = c / (e * g)
        z = acos(h)
        return z

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
        #print(self.state_0)
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

        # target relationship
        self.distance_target = math.hypot(self.state_0.item(0) - self.target_position[0],
                                          self.state_0.item(1) - self.target_position[1]) / 100
        if self.distance_target < 1:
            self.LOS_position = self.target_position
        else:
            self.LOS_position = self.p_f()
        self.distance_LOS = math.hypot(self.state_0.item(0) - self.LOS_position[0],
                                       self.state_0.item(1) - self.LOS_position[1]) / 100
        self.course_LOS = self.desired_course(self.LOS_position[0], self.LOS_position[1],
                                              self.state_0.item(0), self.state_0.item(1))
        self.course_error_LOS = self.state_0.item(3) - self.course_LOS
        # obstacle relationship
        # self.obstacle = env_obstacle
        # self.distance_obstacle = np.zeros((1, 50))[0]
        for i in range(50):
            self.distance_obstacle[i] = (math.hypot(self.state_0.item(0) - self.obstacle[i][0],
                                                    self.state_0.item(1) - self.obstacle[i][1]) - self.obstacle[i][
                                             2]) / 100
        #print(self.distance_obstacle)
        # find the nearest obstacle
        i_obstacle = list(self.distance_obstacle).index(min(self.distance_obstacle))
        self.d1.append(self.distance_obstacle[i_obstacle])

        data_storage(self.x1, self.y1, self.phi1, self.t, self.e1, self.d1, u1 = self.u1, rudder_angle = self.Rudder_angle)  # store data in local files
        self.course_obstacle = self.desired_course(self.obstacle[i_obstacle][0], self.obstacle[i_obstacle][1],
                                              self.state_0.item(0), self.state_0.item(1))
        self.course_error_obstacle = self.state_0.item(3) - self.course_obstacle
        observation = np.array([self.distance_LOS, self.course_error_LOS, self.distance_obstacle[i_obstacle], self.course_error_obstacle, rudder_angle])
        return observation, self.state_0.item(3), self.course_LOS

    def step(self, action, observation):
        rudder_control = observation[-1] + action
        s_, course_real, course_desired = self.obser(rudder_control)
        reach = 0

        if s_[0] < 0.02:
            done = True
            reach = 1
        elif s_[2] <= 0:
            done = True
        else:
            done = False
        return s_, done, reach, course_real, course_desired

    def render(self):

        data_viewer(self.x1, self.y1, u1=self.u1, phit=self.phi1, rudder_angle=self.Rudder_angle, t=self._t, xlim_left=-10, xlim_right=1200, ylim_left=-10, ylim_right=1200,
                        goal_x=1000, goal_y=1000)



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
controllor = PID(5, 6, 0.1, -1, 1, -0.1, 0.1)
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')
course_real = 0
course_desired = 0
step = 0
sess = tf.Session()
#new_saver = tf.train.import_meta_graph('/home/wp/waveglider_RL/Environment/data/Mymodel-100000.meta')
#graph = tf.get_default_graph()
actor = Actor(sess, action_dim, action_bound)
new_saver = actor.new_saver
new_saver.restore(sess, tf.train.latest_checkpoint('C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\Strategy10'))



s, course_real, course_desired = env.reset()
s = s[np.newaxis, :]
while True:
    if step % 10 == 0:
        env.render()
    if s[0][2] > 1:
        a = controllor.update(course_real, course_desired)
    else:
        a = sess.run(actor.a, feed_dict={S: s})[0]
    s_, done, reach, course_real, course_desired = env.step(float(a), s[0])
    if reach == 1:
        success += 1

    s = s_
    s = s[np.newaxis, :]

    if done:
        break
    step += 1
print(success, 'Amazing! Eve give you a kiss!!!')