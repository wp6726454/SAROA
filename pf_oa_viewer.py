import matplotlib.pyplot as plt
import numpy as np
# env_test = np.load("C:\\Users\\user\\PycharmProjects\\waveglider_RL\\Environment\\data\\test_environments\\env_test.npy")

env_test = np.array([
    [106, 754, 14],
    [213, 109, 15],
    [540, 548, 17],
    [178, 436, 11],
    [288, 91, 18],
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
    [819, 857, 13],
    [557, 699, 14],
    [927, 715, 11],
    [327, 567, 12.],
    [400, 368, 16.],
    [189, 184, 10.],
    [710, 700, 13.],
    [738, 835, 15.],
    [1000, 800, 17.],
    [358, 178, 18.],
    [535, 264, 14.],
    [500, 451, 15.],
    [301, 453, 16.],
    [690, 419, 19.],
    [59, 53, 12.],
    [941, 908, 14.],
    [150, 500, 18.],
    [74, 258, 18.],
    [550, 163, 16.],
    [343, 597, 16.],
    [283, 223, 14.],
    [400, 1000, 17.],
    [156, 46, 18.]])
pointsway = np.array([[501, 500], [1000, 1000]])

def data_viewer(x1, y1, u1, phit, rudder_angle, t,
                xlim_left=-100, xlim_right=1200,ylim_left=-100, ylim_right=1200,
                goal_x=100, goal_y=100,
                T='', Ffoil_x='', obs_x=0, obs_y=0, obs_R=0):
    plt.figure(1, figsize=(10, 10))
    # grid = plt.GridSpec(3, 2, wspace=0.5, hspace=0.5)
    # path = plt.subplot(grid[0:3, 0])
    plt.plot(y1, x1, label='Sailing path', color='b')
    #set_title('Path')
    plt.axis([ylim_left, ylim_right, xlim_left, xlim_right])
    theta = np.arange(0, 2*np.pi, 0.01)
    # if len(y1) > 1:
    #     plt.plot(y1[-1] + 100 * np.cos(theta), x1[-1] + 100 * np.sin(theta), color='g')  # USV domain
    plt.plot(1000, 990, color='b')  # start position
    # plt.plot(400+1*np.cos(theta), 500+1*np.sin(theta), color='r')  # way_point
    # plt.plot(800+1*np.cos(theta), 600+1*np.sin(theta), color='r')  # way_point
    plt.plot(0+2*np.cos(theta), 0+2*np.sin(theta), color='r')  # target position
    for i in range(len(pointsway)-1):  # track lines
        plt.plot([pointsway[i][0], pointsway[i+1][0]], [pointsway[i][1], pointsway[i+1][1]], color='r')
    # Obstacle development
    for i in range(50):
        plt.plot(env_test[i][1] + env_test[i][2] * np.cos(theta), env_test[i][0] + env_test[i][2] * np.sin(theta), color='k')

    # plt.set_ylabel('x(m)')
    # plt.set_xlabel('y(m)')
    # plt.legend()

    # heading = plt.subplot(grid[0, 1])
    # heading.plot(t, phit, '-r', label='phit')
    # heading.set_title('Heading')
    # heading.set_ylabel('Course(rad)')
    #
    # speed_plot = plt.subplot(grid[1, 1])
    # speed_plot.plot(t, u1, '-r')
    # speed_plot.set_title('Sailing speed')
    # speed_plot.set_ylabel('Speed(m/s)')
    #
    # rudder_angle_plot = plt.subplot(grid[2, 1])
    # rudder_angle_plot.plot(t, rudder_angle, '-r')
    # rudder_angle_plot.set_title('Rudder angle')
    # rudder_angle_plot.set_ylabel('Rudder angle(deg)')
    # rudder_angle_plot.set_xlabel('Time(s)')


    plt.pause(0.00001)


