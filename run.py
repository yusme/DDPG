import numpy.random as nr
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import tensorflow as tf
import time
import filter as fil
import DDPG as ddpg
import numpy as np
import gym
import os
import pylab as pyl


num_episode=500
num_experience=4
total_state =500

MemorySize=100000
start=5000
batch_size=32


replay_transition=100

test=1
initial= True

Mean_reward=[]
max_reward=[]
min_reward=[]
Mean_q=[]


print ("version Tensorflow is")
print(tf.__version__)



def main():


    env    = gym.make('Pendulum-v0');
    env_ren = gym.make('Pendulum-v0');
    current_state = env.reset();
    #env.monitor.start('tmp/Pendulum-v0-experiment-1')

    validation_env(env)



    newEnv     = fil.makeFilteredEnv(env);
    act_dim    = newEnv.action_space.shape[0];
    obs_dim    = newEnv.observation_space.shape[0];



    print ("DDPG REINFORCEMENT LEARNING ")



    print "env" ,env.action_space.shape ," and  " ,[1]+list(newEnv.action_space.shape), act_dim
    print "env", env.observation_space.shape, " and  ", newEnv.observation_space.shape, obs_dim
    env = newEnv;

    session = tf.Session();
    session.run(tf.initialize_all_variables())

    print "ini", session.run(tf.initialize_all_variables())
    #saver = tf.train.Saver()
    #writer = tf.train.SummaryWriter("/tmp/mnist_logs", session.graph_def)
    #saver.restore(session, "tmp/model.ckpt")


    # - Initialize the agent-
    agent = ddpg.DDPG(newEnv)
    avg_reward = []
    after = 0



    print("---avarage_reward---\n", len(avg_reward))

    start_time = time.time()
    session.run(tf.initialize_all_variables())

    for i_episode in xrange(num_episode):

        print "-----------\n Episode=", i_episode,"/",num_episode,
        current_state = env.reset()
        env.step(env.action_space.sample())
        # env.reset()

        action = env.action_space.sample()#agent.get_action(session, current_state)
        #action = action + noise_act(act_dim)


        for t in xrange(total_state):#1000 total state
            if  agent.get_buffer_size() > start:
                agent.q_actor_critic(session, batch_size)
                if t % 1000 == 0:
                    print "----", t

            else:

                next_state, reward, done, info = env.step(action[0])

                action = agent.get_action(session, next_state)
                action = action + agent.noise_act(action.shape)
                priotity= 1

                experience = [current_state, action, reward, next_state, done, info,priotity]
                agent.replay_buffer.add(experience)

                current_state = next_state

                if done:
                    break


        print ("\n  apply_policy_gradient")
        agent.apply_policy_gradient(session, 62)



        print ("\n start test")
        current_state = env.reset()
        test_env(env,action,agent,session,avg_reward)



        # Save the variables to disk.
        #save_path = saver.save(session, "tmp/model.ckpt")
        #print("Model saved in file: %s" % save_path)

        if(i_episode%100==0 and i_episode!=1):
            plot_reward()
            plot_perfomance(avg_reward)




def test_env(env,action, agent,session,avg_reward):
    rewards=[]
    q_value = []
    values=[]
    Z=[]

    for t in xrange(test):

        total_reward = 0

        current_state = env.reset()

        for j in range(100):
            env.render()

            action = agent.get_action(session, current_state)
            #q_v= agent.critic.get_Q_Value_critic( session,current_state[0],action)
            #q_value.append(q_v)ne
            current_state, reward, done, info = env.step(action[0])
            #print "current state",current_state

            rewards.append(reward)
            if done:
                print "done bei j=", j

        print "done"
        env.monitor.close()
        print "--Mean reward   ",np.mean(rewards)," ", np.max(rewards)," ",np.min(rewards)

        Mean_reward.append(np.mean(rewards))
        max_reward.append(np.max(rewards))
        min_reward.append(np.min(rewards))
        print("---Avarage_reward---", np.mean(rewards))



def plot_policy(agent,session,episode):
    x = np.arange(-1, 1, 0.025)
    y = np.arange(-1, 1, 0.025)
    X, Y   = pyl.meshgrid(x, y)
    sample = np.ones([80, 80])
    a = []

    print "sample", sample

    for i in range(len(X)):
        for j in range(len(X)):
            a.append([X[i][j], Y[i][j], X[i][j]])

    sample = agent.get_sample_action(session, a)
    sample = agent.get_critic_Q(session, a, sample)
    sample = np.reshape(sample, [80, 80])
    print "---size a", sample.shape
    plot_function_countur2D(-1, 1.0, sample, episode)
    plot_function_countur3D(-1, 1.0, sample, episode)



def validation_env( env ):

    act_space = env.action_space;
    obs_space = env.observation_space;

    if not type(act_space) == gym.spaces.box.Box:
        raise RuntimeError('Environment is no continous action space ');
    if not type(obs_space) == gym.spaces.box.Box:
        raise RuntimeError('Environment is no continous observation space ');


def noise_act(act_dim):
    mu = 0;
    sigma = 0.2;
    theta = 0.15;
    x = np.ones(act_dim) * mu;

    dx = theta * (mu - x) + sigma * nr.randn(len(x))
    noise = x + dx
    return noise



def plot_reward():

    l=len(Mean_reward)
    x = np.arange(l)

    print "Mean Reward ", Mean_reward
    #plt.subplots(2, 1)
    plt.figure()
    plt.plot(x, Mean_reward, 'b-')
    plt.ion()
    plt.show(False)
    time.sleep(1)
    plt.draw()

    #plt.plot(x, max_reward,'b--')
    #plt.plot(x, min_reward,'b--' )
    #plt.plot(x, Mean_q,    'g--')



def plot_perfomance(num):

    l = len(num)
    x = np.arange(l)
    plt.figure(2)
    plt.ion()
    plt.plot(x, num, 'bo')
    plt.show()
    time.sleep(1)


def z_function(x,y):
    return ((x**2+y**3)*np.exp(-(x**2+y**1)/2))

from mpl_toolkits.mplot3d import Axes3D

def plot_function_countur2D(min,max,action,episode):
    x = np.arange(min, max, 0.025)
    y = np.arange(min, max, 0.025)

    # meshgrid create a matrix von min to max with 0.1 as distance
    # the output is 2 matrix where one is the transport of the other one
    X, Y = pyl.meshgrid(x, y)
    # Z=z_function(X,np.transpose(X))

    Z = action

    print Y


    # ==========
    #  2D contur
    # ==========
    plt.figure()
    im = plt.imshow(action, cmap=plt.cm.RdBu)
    cset = plt.contour(action, x, linewidths=2, cmap=plt.cm.Set2)
    plt.colorbar(im)

    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

    plt.ion()
    plt.show(False)
    time.sleep(2)
    initial = False
    plt.draw()

    name_file='exp/episode2D'+str(episode) +'.png'
    print name_file.__str__()
    plt.savefig(name_file)
    plt.close()


def plot_function_countur3D(min,max,action,episode):
    x=np.arange(min,max,0.025)
    y=np.arange(min,max,0.025)
    # meshgrid create a matrix von min to max with 0.1 as distance
    # the output is 2 matrix where one is the transport of the other one
    X, Y = pyl.meshgrid(x, y)
    #Z=z_function(X,np.transpose(X))
    Z=action
    print Y
    #==========
    #  3D
    #==========
    fig=plt.figure()
    #plt.ion()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Y, X, Z, rstride=1, cstride=1,
                           cmap=plt.cm.RdBu, linewidth=0, antialiased=False)
    ax.set_xlim([min, max])
    ax.set_ylim([min, max])
    ax.zaxis.set_major_locator(plt.LinearLocator(3))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show(False)
    plt.draw()
    name_file = 'exp/episode3D' + str(episode) + '.png'
    print name_file.__str__()
    plt.savefig(name_file.__str__())
    plt.close()


    # ==========
    #  2D contur
    # ==========

    #im = plt.imshow(action, cmap=plt.cm.RdBu)
    #cset = plt.contour(action, x, linewidths=2, cmap=plt.cm.Set2)
    #plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    #plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    #plt.colorbar(im)


    #fig=plt.figure(1)
    #plt.ion()
    #plt.show(False)
    ##time.sleep(2)
    #initial=False

    #plt.draw()
    #time.sleep(2)
    #plt.close()



if __name__ == '__main__':
   main()





   # env.monitor.close()

   # filter = fil.filter(env);

   # self.env = filter_env.makeFilteredEnv()

   # print all variable into Enviroment

   # Create agend DDPG and gob per paramter  obs and action

   # run a for with total training time

   # run a for with