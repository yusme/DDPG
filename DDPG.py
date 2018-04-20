import numpy.random as nr
import Critic_network as Critic
import actor_netwok   as Actor
import replay as replayMemory
import tensorflow as tf
import numpy as np
import gym



Minibatch_size=20
MemorySize=100000
Gamma = 0.99

mu = 0;
sigma = 0.2;
theta = 0.15;
Mean_TD=30;

learning_rate=1e-4



class DDPG:


    def __init__(self, env):


        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        print "------env ",env.observation_space.shape[0], " " ,self.act_dim, " obs ",env.action_space.shape[0],self.obs_dim
        action_bound = env.action_space.high

        self.avarage_reward=[]
        self.loss=[]

        self.noise_action=self.noise_act(self.act_dim)

        #self.add_noise=self.noise_action(self.act_dim);

        # ---------------------------
        # - Initialize  Replay Memory
        #  ---------------------------
        # q-learning is off policy algorithmus
        #  since it update the q-value with put
        # making assumption about the actual policy


        self.replay_buffer = replayMemory.Memory( MemorySize, Minibatch_size, self.act_dim,self.obs_dim )

        # ---------------------------
        # - Initialize  Q Learning with function approximation,
        # ---------------------------

        # the critic Produce a TD temporal difference and is update from
        # the gradient obtain from the td error signal

        self.critic= Critic.CriticNetwork(self.act_dim, self.obs_dim, Minibatch_size )
        self.actor_policy = Actor.PolicyNetwork(self.obs_dim, self.act_dim, Minibatch_size)

        # session = tf.Session();
        session = tf.InteractiveSession()
        session.run(tf.initialize_all_variables())
    def get_buffer_size(self):
        return self.replay_buffer.containerSize

    def get_actions(self, batchsize):
        experiences = self.replay_buffer.select_sample(batchsize)
        return experiences[1]

    def get_action(self,session, current_state):
        action= self.actor_policy.get_action(session,current_state);
        return action

    def get_sample_action(self, session, current_state):
        actions = self.actor_policy.get_sample_action(session, current_state)
        return actions

    def get_critic_Q(self, session, obs, action):
        return  self.critic.get_Q_Value_critic(session,obs, action)



    def get_target(self, session, experiences):

        rewards = experiences[2]
        done    = experiences[4]
        next_observation = experiences[3]
        y = np.zeros([len(rewards),1])

        target_actions   = self.actor_policy.get_sample_target_action(session, next_observation)
        q_values_targets = self.critic.get_target_Q_value(session, next_observation, target_actions  )

        for k in range(len(done)):
            if done[k]:
                print "entro en done",done[k]
                y[k]=rewards[k]
            else:
                y[k] = rewards[k] + Gamma * q_values_targets[k]

        return y



    def apply_policy_gradient(self, session,batchsize):
        experiences = self.replay_buffer.select_sample(batchsize)

        observations = experiences[0]
        actions      = experiences[1] + self.noise_act(self.act_dim)


        critic_gradient = self.critic.apply_gradient(session,observations, actions)
        self.actor_policy.apply_policy_gradient(session,observations,critic_gradient[0])
        self.actor_policy.update_target(session)




    def q_actor_critic(self, session, batchsize):

        experiences = self.replay_buffer.select_sample(batchsize)



        observations = experiences[0]
        actions      = experiences[1]

        actions = actions + self.noise_act(self.act_dim)

        y = self.get_target(session, experiences)

        self.q_state_action= self.critic.get_Q_Value_critic(session,observations, actions)



        self.critic.optimize_critic_net(session, y ,observations, actions)


        Mean_TD_1=np.mean(np.square(y- self.q_state_action))


        #print "minimun" ,Mean_TD_1


        self.critic.update_target(session)

        #self.actor_policy.update_target(session)




    def noise_act(self,act_dim):
        mu = 0;
        sigma = 0.2;
        theta = 0.15;
        x = np.ones(act_dim) * mu;

        dx = theta * (mu - x) + sigma * nr.randn(len(x))
        noise = x + dx
        return noise

    def get_performance(self):
        return [self.avarage_reward, self.loss]
