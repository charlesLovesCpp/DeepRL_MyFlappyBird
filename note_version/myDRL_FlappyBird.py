from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")

import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000.
EXPLORE = 2000000.
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding="SAME")

def createNetwork():
    # network weights
    # 图像处理层

    # 卷积大小为8x8,一次处理连续的4张,卷积深度为32
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    # 决策层

    # 全连接层 输入为1600个节点,输出为512个节点
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([ACTIONS])

    # 全连接层 输入为512个节点,输出为2个节点
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layer
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 注意了,下面两个pool都没有用到
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])

    # 注意了,这里把3*3*64=576个节点铺成了1600个节点
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost funtion
    # 训练的是 target Q
    # 误差函数是一系列动作a与readout相乘后,每行的值加起来的Q值,再与y求均方差
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game stae to communicate with emulator
    game_state = game.GameState()

    # store the previous observatios in replay memory
    D = deque()

    # printing
    a_file = open("logs_"+ GAME + "/readout.txt", 'w')
    h_file = open("logs_"+ GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # 返回第一张图片为初始点,变换大小并转成灰度图
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # 将灰度大于1的点转成255,白色,小于的变成0,黑色
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # 把四张同样的初始图像堆叠成4个,作为初始输入
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("state_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        # readout是神经网络的输出值,每个行动的Q值
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        # 这里定义是每帧动一次
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                # 找到能得到最高Q的那个行动
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE


        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # 保留s_t的前三个图像,在最前面加上新图像x_t1,有点像个滑动窗口
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        # 上一状态,现在的动作,动作的reward,到达的下一状态, 是否结束了
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            
            # get the batch variables
            # 把minibatch各个维数的值赋给不同的向量
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            # 原理:
            # 本质还是迭代更新,用样本数据来更新新的y_batch,然后再拿新的y_batch
            # 去重新训练神经网络,让y_batch越来越真实,这样最后会得到一个收敛的y_batch
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal,  only equals reward
                if teminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append([r_batch[i] + GAMMA * np.max(readout_j1_batch[i])])

            # perform gradient step
            train_step.run(feed_dict={y: y_batch,
                                      a: a_batch,
                                      s: s_j_batch})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION", action_index,
              "/ REWARD", r_t, \
              "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t])+ '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]])+ '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''





def playGame():
    sess = tf.InteractiveSession()
    # 创建深度网络
    s, readout, h_fc1 = createNetwork()
    # 开始训练
    trainNetwork(s, readout, h_fc1, sess)
    sess.close()

def main():
    playGame()

if __name__ == "__main__":
    main()
