from random import sample as random
import numpy as np
import collections as memory
import random

alpha=0.7
beta=0.5

class Memory:

     def __init__(self,MemorySize, batch_size, act_dim,obs_dim):
         self.Memorysize = MemorySize
         self.batch_size = batch_size
         self.container= memory.deque()
         self.containerSize = 0
         self.priority=1
         self.act_dim=act_dim
         self.obs_dim=obs_dim



     def get_size(self):
         return self.batch_size

     def select_batch(self, batchSize):
         return random.sample(self.container, batchSize)

     def clear_memory(self):
         self.container = memory.deque()
         self.num_experiences = 0

     def add(self, experience):
         #experience = [current_state, action, reward, next_state, done, info]
         #print "-------------experience \n "," cu_state \n", experience[0],"\n action \n", experience[1],"\n", experience[4]

         experience.append(self.priority)
         if self.containerSize < self.Memorysize:
            self.container.append(experience)
            self.containerSize = self.containerSize+1
            #print "self.containerSize3333", self.containerSize, self.Memorysize
         else:
             self.container.popleft()
             self.container.append(experience)


     #def computer_priority(self):
         #priority=

     def select_sample_priority(self, batch_size):

         sample = random.sample(self.container, batch_size)

         current_state = [x[0] for x in sample]
         actions = np.asarray([x[1] for x in sample])
         rewards = [x[2] for x in sample]
         next_state = [x[3] for x in sample]
         done = [x[4] for x in sample]
         priority = [x[5] for x in sample]

         # [current_state, action, reward, next_state, done, info]
         obs_dim = self.obs_dim

         act_dim = self.act_dim

         current_state = np.resize(current_state, [batch_size, obs_dim])
         actions = np.resize(actions, [batch_size, act_dim])
         rewards = np.resize(rewards, [batch_size, act_dim])
         next_state = np.resize(next_state, [batch_size, obs_dim])
         done = np.resize(done, [batch_size, act_dim])
         priority =  np.resize(priority, [batch_size, act_dim])
         #

         # print "           current state \n", actions.shape, current_state.shape,
         # print "action  \n",actions[0],actions[1],actions[2]
         # print "current state  \n", current_state[0], current_state[1], current_state[2]
         # print " reward\n\n", rewards,"reward shape" ,rewards.shape
         # print " -------done\n\n", done, "done shape", done.shape

     def prioritized_replay(self, session, batchsize, writer):

         N = MemorySize
         beta = 0.5
         alpha = 0.5

         experiences = self.replay_buffer.select_sample_priority(batchsize)
         observations = experiences[0]
         actions = experiences[1]
         priority = experiences[6]

         w = np.ones(len(priority))
         print "weigth", w

         # ------Computer TD error--------

         y = self.get_target(session, experiences)
         q = self.critic.get_Q_Value_critic(session, observations, actions)

         critic_gradient = self.critic.apply_gradient(session, observations, actions)

         td_error = np.absolute(np.square(y - q))

         # important_sampling
         w = np.power((N * priority), beta)
         w = np.divide(w, max(w))
         sum(priority)
         priority = td_error
         accu_w = accu_w + (w[k] * td_error[k] * critic_gradient)

         theta = self.critic
         theta = theta + (learning_rate * accu_w)






     def select_sample(self,batch_size):
        # print "batch_size",batch_size



         sample=random.sample(self.container, batch_size)


         current_state=  [x[0] for x in sample]
         actions =     np.asarray([x[1] for x in sample])
         rewards =     [x[2] for x in sample]
         next_state=   [x[3] for x in sample]
         done =        [x[4] for x in sample]


         #[current_state, action, reward, next_state, done, info]
         obs_dim=self.obs_dim

         act_dim = self.act_dim

         current_state = np.resize(current_state,[batch_size,obs_dim])
         actions       = np.resize(actions, [batch_size, act_dim])
         rewards       = np.resize(rewards, [batch_size, act_dim])
         next_state    = np.resize(next_state, [batch_size, obs_dim])
         done          = np.resize(done, [batch_size, act_dim])
         #

         #print "           current state \n", actions.shape, current_state.shape,
         #print "action  \n",actions[0],actions[1],actions[2]
         #print "current state  \n", current_state[0], current_state[1], current_state[2]
         #print " reward\n\n", rewards,"reward shape" ,rewards.shape
         #print " -------done\n\n", done, "done shape", done.shape



        # observations=np.reshape(observations, [batch_size * obs_dim, obs_dim])
         #actions=np.reshape(actions, [(batch_size * 3) , act_dim])


         return current_state, actions, rewards, next_state,done;












