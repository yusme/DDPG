import tensorflow as tf
import numpy as np
import math


Num_layers1 = 400;
Num_layers2 = 300;
learning_rate=1e-4;
Tau = 0.001


#TODO: summary tensorboard


class PolicyNetwork:

    def __init__(self,obs_dim, act_dim, minibatch):
        self.obs_dim = obs_dim;
        self.act_dim = act_dim;


        # Create policyNetwork

        self.action_policy, self.theta, self.observation=\
            self.create_PolicyNetwork(obs_dim,
                                      act_dim)

        # Create TargetNetwork

        self.act_output_target,self.newTarget,self.target_theta,self.state = \
            self.create_target_network( obs_dim,
                                        act_dim,
                                        self.theta)

        self.define_optimizer_network()




        # Merge all the summaries and write them out to /tmp/mnist_logs
       # merged = tf.merge_all_summaries()


    def randomUniform(self, shape, f):
        # f = f or shape[0]
        nmin = -1 / np.sqrt(f)
        nmax = 1 / np.sqrt(f)

        return (tf.random_uniform(shape, minval=nmin, maxval=nmax))



    def summary(self,parameter):
        for par in parameter:
            tf.histogram_summary(par.name,par)



    def initialize_parameter(self, obs_dim, act_dim):
        print "act dim**", act_dim,obs_dim

        w1 = tf.Variable(self.randomUniform([obs_dim, Num_layers1], obs_dim))
        b1 = tf.Variable(self.randomUniform([Num_layers1], obs_dim))
        w2 = tf.Variable(self.randomUniform([Num_layers1, Num_layers2], Num_layers1))
        b2 = tf.Variable(self.randomUniform([Num_layers2], Num_layers1))

        w3 = tf.Variable(tf.random_uniform([Num_layers2, act_dim], 3e-3))
        b3 = tf.Variable(self.randomUniform([act_dim], 3e-3))
        print "w3=",w3.get_shape()
        print "b3=", b3.get_shape()

        #w3 = tf.Variable(tf.random_uniform([Num_layers2, act_dim], -3e-3, 3e-3));
        #print "w3=",w3.get_shape()
        #b3 = tf.Variable(tf.random_uniform([act_dim], -3e-3, 3e-3))
       # print "b3=", b3.get_shape()


        theta=[w1,b1,w2,b2,w3,b3]

        #self.summary(theta)

        return theta



    def create_PolicyNetwork(self, obs_dim, act_dim):

         observation = tf.placeholder("float", [None, obs_dim])

         [w1, b1, w2, b2, w3, b3]=\
             self.initialize_parameter ( obs_dim, act_dim)

         layer1 = tf.nn.relu(tf.matmul(observation, w1) + b1)
         layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

         action_output = tf.tanh(tf.matmul(layer2, w3) + b3)

         #scaled_out = tf.mul(action,action_bound)


         return action_output,[w1,b1,w2,b2,w3,b3], observation



    def create_target_network(self, obs_dim, act_dim, theta):
        # Low pass Filter exponential movement avarage
        # to stabilize the learning

        state = tf.placeholder("float", [None, obs_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - Tau)
        newTarget    = ema.apply(theta)
        target_theta = [ema.average(i) for i in theta]

        layer1     = tf.nn.relu(tf.matmul(state, target_theta[0]) + target_theta[1])
        layer2     = tf.nn.relu(tf.matmul(layer1, target_theta[2]) + target_theta[3])
        act_output = tf.tanh(tf.matmul(layer2, target_theta[4]) + target_theta[5])

        #self.summary(target_theta)

        return act_output, newTarget, target_theta, state




    def define_optimizer_network(self):
        self.q_gradient = tf.placeholder("float", [None, self.act_dim])
        weight_decay    = tf.add_n([0.01 * tf.nn.l2_loss(x) for x in self.theta])
        self.tf_gradients = tf.gradients(self.action_policy, self.theta, -self.q_gradient)
        zip_grad          = zip(self.tf_gradients, self.theta)
        self.optimizer    = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip_grad)




    def apply_policy_gradient(self, session, observations, q_gradient):
        #print "Critic gradient222 =\n",  q_gradient,"\n",
        feed_dict = {self.observation: observations,
                     self.q_gradient:q_gradient
                    }

        return session.run(self.optimizer, feed_dict)


    def get_action(self, session, current_state):
        feed_dict = {self.observation: [current_state]}

        return session.run(self.action_policy, feed_dict)


    def get_target_action(self, session, current_state):
        feed_dict = {self.observation: [current_state]}
        return session.run(self.act_output_target, feed_dict)


    def get_sample_action(self, session, current_state):
        feed_dict = {self.observation: current_state}
        return session.run(self.action_policy, feed_dict)

    def get_sample_target_action(self, session, current_state):
        feed_dict = {self.observation: current_state}
        return session.run(self.action_policy, feed_dict)

    def update_target(self, session):
       # session.run(self.newTarget);
        weights = self.theta
        target_weights = self.target_theta
        for i in xrange(len(weights)):
            target_weights[i] = Tau * weights[i] + target_weights[i] * (1 - Tau)
            #print "self.target_theta",target_weights[i].eval
        self.target_theta = target_weights




        #for i in xrange(len(weights)):
            #target_weights[i] = Tau * weights[i] + target_weights[i] * (1 - Tau)

        #self.target_theta = target_weights

        #self.update_target_network_params = \
        #    [self.target_network_params[i].assign(
        #        tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
        #     for i in range(len(self.target_network_params))]




    def create_Batch_normalization_PolicyNetwork(self, obs_dim, act_dim):

        # Note that pre-batch normalization bias is ommitted. The effect of this bias would be
        # eliminated when subtracting the batch mean. Instead, the role of the bias is performed
        # by the new beta variable. See Section 3.2 of the BN2015 paper.

        [w1, b1, w2, b2, w3, b3] = self.initialize_parameter(observation, obs_dim, act_dim)

        scale1 = tf.Variable(self.randomUniform([Num_layers1], obs_dim), name='scale1')
        beta1 = tf.Variable(self.randomUniform([Num_layers1], obs_dim), name='beta1')
        scale2 = tf.Variable(self.randomUniform([Num_layers1], obs_dim), name='scale2')
        beta2 = tf.Variable(self.randomUniform([Num_layers1], obs_dim), name='beta2')

        observation = tf.placeholder("float", [None, obs_dim])
        epsilon = 1e-3

        # Calculate batch mean and variance
        layers1 = tf.matmul(observation, w1)
        batch_mean, batch_var = tf.nn.moments(layers1,[0])

        # Apply batch normalizing
        transf_L1  = (layers1-batch_mean)/ tf.sqrt(batch_var+epsilon)
        layers1 = tf.nn.relu( scale1* transf_L1 + beta1)

        layers2 = tf.matmul(layers1, w2)
        batch_mean_L2, batch_var_L2 = tf.nn.moments(layers2,[0])

        # Apply batch normalizing
        norm = tf.nn.batch_normalization(layers2,batch_mean_L2, batch_var_L2, beta2,scale2, epsilon)
        layers2 = tf.nn.relu(norm)

        action = tf.nn.tanh(tf.matmul(layers2, w3) + b3);

        return action, [w1, b1, w2, b2, w3, b3], observation



        #loss = tf.reduce_mean(tf.square(y-ydata))
       # optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer.compute_gradients(loss, self.theta)
            #tf.train.GradientDescentOptimizer(0.5)
        #train = optimizer.minimize(loss)



        # the actor computer the gradient of the expectete reward

        #correct_predition = tf.equal(self.y_pred_cls, self.y_true_cls)
        #accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers3, labels=self.y_true)
        #cost = tf.reduce_mean(self.theta, 0)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients()
        #return "action"











