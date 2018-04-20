import tensorflow as tf
import numpy as np

Weight_decay_rate=0.01
learning_rate=1e-4
unitsL1 =  400
unitsL2 =  300
Tau     = 0.001


class CriticNetwork:

#TODO update wiegth decay


    def __init__(self, act_dim, obs_dim, minibatch):

        self.observation = tf.placeholder("float", [None, obs_dim])  # tf.identity(obs_dim, name='h0-obs')
        self.action      = tf.placeholder("float", [None, act_dim])  # tf.identity(act_dim, name='h0-act')

        self.q_output, self.theta_Q_network = self.create_Q_Network(act_dim,
                                                                    obs_dim)
        self.q_output_target,self.newTarget,self.target_theta = \
            self.create_target_network(act_dim,
                                       obs_dim,
                                       self.theta_Q_network)

        self.optimizer,self.gradient_Q_network = \
            self.define_optimizer( obs_dim, act_dim)


    # definition initialize the Parameter

    def randomUniform(self, shape, f):
        max = 1 / np.sqrt(f)
        min = -max
        return (tf.random_uniform(shape, min, max));



    def define_weights_biases(self, obs_dim, act_dim):

        # ---- theta variable--------

        w1 = tf.Variable(self.randomUniform([obs_dim, unitsL1], obs_dim))
        b1 = tf.Variable(self.randomUniform([unitsL1], obs_dim))
        b3 = tf.Variable(tf.random_uniform([1], -3e-4, 3e-4));
        w2 = tf.Variable(self.randomUniform([unitsL1, unitsL2], unitsL1 + act_dim))
        w2_act = tf.Variable(self.randomUniform([act_dim, unitsL2], unitsL1 + act_dim))
        b2 = tf.Variable(self.randomUniform([unitsL2], unitsL1 + act_dim))

        w3 = tf.Variable(tf.random_uniform([unitsL2, 1], -3e-4, 3e-4));


        theta = [w1, b1, w2,w2_act, b2, w3, b3]

        return theta



    def create_Q_Network(self, act_dim, obs_dim):


        [w1, b1, w2, w2_act, b2, w3, b3] = self.define_weights_biases(obs_dim, act_dim)

        layers1 = tf.nn.relu( tf.matmul(self.observation,w1) +b1 );
        layers2 = tf.nn.relu( (tf.matmul(layers1,w2)+ tf.matmul(self.action,w2_act) ) +b2 );
        q_Value = tf.nn.tanh(tf.matmul(layers2,w3)+b3);

        theta = [w1, b1, w2, w2_act, b2, w3, b3]

        return q_Value, theta


    def create_target_network(self, act_dim,  obs_dim, theta_Q_net ):

        self.obs_target = tf.placeholder("float", [None, obs_dim])  # tf.identity(obs_dim, name='h0-obs')
        self.act_target = tf.placeholder("float", [None, act_dim])  # tf.identity(act_dim, name='h0-act')


        ema       = tf.train.ExponentialMovingAverage(decay=(1 - Tau))
        newTarget = ema.apply(theta_Q_net)
        target_theta = [ema.average(i) for i in theta_Q_net]

        [w1, b1, w2,w2_act, b2, w3, b3] = target_theta

        layers1  = tf.nn.relu(tf.matmul(self.obs_target, w1) + b1)
        layers2 = tf.nn.relu((tf.matmul(layers1, w2) + tf.matmul(self.act_target, w2_act)) + b2);
        q_output_target = tf.tanh(tf.matmul(layers2, w3) + b3)

        return q_output_target, newTarget, target_theta




    #def get_weight_decay(self):



    def define_optimizer(self,obs_dim,act_dim ):
        gradient_Q_network = tf.gradients(self.q_output, self.action)
        weight_decay= tf.add_n([Weight_decay_rate * tf.nn.l2_loss(x) for x in self.theta_Q_network])
        loss = tf.reduce_mean(  tf.square(  self.q_output_target- self.q_output) )+weight_decay
        optimizer   = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer,gradient_Q_network


    def optimize_critic_net(self,session, q_target, observations, actions):
        feed_dict = {self.action: actions,
                     self.observation: observations,
                     self.q_output_target:q_target}
        return session.run( self.optimizer, feed_dict)


    def apply_gradient(self, session,observations, actions ):
        feed_dict = {self.action: actions,
                     self.observation: observations }
        return session.run(self.gradient_Q_network,feed_dict)


    def get_action(self, session, action_batch, observation_batch):
        feed_dict = {self.action: action_batch,
                     self.observation: observation_batch}
        return session.run(self.q_output, feed_dict=feed_dict)

    def get_target_Q_value(self,session, observation, action):
        feed_dict = {self.obs_target:observation,
                     self.act_target:action}
        return session.run(self.q_output_target, feed_dict)

    def get_Q_Value_critic(self,session, observations, actions):
        feed_dict = {self.observation: observations,
                     self.action: actions}
        return session.run(self.q_output, feed_dict)


    def update_target(self, session):
        #session.run(self.newTarget);

        weights = self.theta_Q_network
        target_weights = self.target_theta
        for i in xrange(len(weights)):
            target_weights[i] =   Tau * weights[i] + target_weights[i] * (1 - Tau)
        self.target_theta = target_weights


    def get_theta_Q_network(self):
        return self.theta_Q_network














