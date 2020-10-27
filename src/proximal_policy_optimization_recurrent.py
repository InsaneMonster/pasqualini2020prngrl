# Import packages

import tensorflow
import numpy
import scipy.signal
import math
import random

# Import required src

from usienarl import SpaceType, Config, Model
from usienarl.utils.common import softmax


class Buffer:
    """
    A buffer working the same way of the default PPO one but with added storage for LSTM states.
    """

    def __init__(self,
                 parallel_amount: int,
                 discount_factor: float, lambda_parameter: float):
        # Define buffer components
        self._observations: [] = []
        self._actions: [] = []
        self._rewards: [] = []
        self._values: [] = []
        self._log_likelihoods: [] = []
        self._last_step_rewards: [] = []
        self._lstm_states: [] = []
        # Define parameters
        self._discount_factor: float = discount_factor
        self._lambda_parameter: float = lambda_parameter
        self._parallel_amount: int = parallel_amount
        # Define buffer pointer attributes
        self._end_trajectories_pointers: [] = []
        self._pointer: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=int)
        self._episode_done_previous_step: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=bool)

    def store(self,
              observation: numpy.ndarray,
              action: numpy.ndarray,
              reward: numpy.ndarray,
              episode_done: numpy.ndarray,
              value: numpy.ndarray,
              log_likelihood: numpy.ndarray,
              lstm_state: numpy.ndarray):
        """
        Store the time-step in the buffer, taking care of the parallelization of the environment.

        :param observation: the current observation to store in the buffer wrapped in a numpy array
        :param action: the last action to store in the buffer wrapped in a numpy array
        :param reward: the reward obtained from the action at the current state to store in the buffer wrapped in a numpy array
        :param episode_done: the flag defining the end of the episode at the current state to store in the buffer wrapped in a numpy array
        :param value: the value of the state as estimated by the value stream of the model to store in the buffer wrapped in a numpy array
        :param lstm_state: the LSTM network state before taking the stored action
        :param log_likelihood: the log likelihood of the action on the state as estimated by the model wrapped in a numpy array
        """
        # Append all data and increase the pointer
        self._observations.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._log_likelihoods.append(log_likelihood)
        self._lstm_states.append(lstm_state)
        # Update pointer by adding one where the episode is not already finished
        self._pointer = numpy.where(numpy.logical_not(self._episode_done_previous_step), self._pointer + 1, self._pointer)
        # Update the stored episode done flags
        self._episode_done_previous_step = episode_done.copy()

    def get(self) -> []:
        """
        Get all of the data from the buffer, serializing all the parallel data.
        Also reset pointers in the buffer and the lists composing the buffer.

        :return a list containing the numpy arrays of: observations, actions, advantages, rewards-to-go, log-likelihoods, LSTM states
        """
        # Generate numpy arrays out of the buffer
        observations_array: numpy.ndarray = numpy.stack(numpy.array(self._observations))
        actions_array: numpy.ndarray = numpy.stack(numpy.array(self._actions))
        log_likelihoods_array: numpy.ndarray = numpy.stack(numpy.array(self._log_likelihoods))
        lstm_states_array: numpy.ndarray = numpy.stack(numpy.array(self._lstm_states))
        # Prepare serialized list of all data in the buffer
        serialized_observations: [] = []
        serialized_actions: [] = []
        serialized_log_likelihoods: [] = []
        serialized_lstm_states: [] = []
        advantages: [] = []
        rewards_to_go: [] = []
        trajectory_start_pointer: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=int)
        for trajectory_index in range(len(self._end_trajectories_pointers)):
            pointer: numpy.ndarray = self._end_trajectories_pointers[trajectory_index]
            last_step_reward: numpy.ndarray = self._last_step_rewards[trajectory_index]
            for i in range(self._parallel_amount):
                # Get the trajectory slice for the current parallel episode
                trajectory_slice = slice(trajectory_start_pointer[i], pointer[i])
                # Compute rewards and values by appending the last step reward
                rewards: numpy.ndarray = numpy.array(numpy.array(self._rewards)[trajectory_slice, i].tolist() + [last_step_reward[i]])
                values: numpy.ndarray = numpy.array(numpy.array(self._values)[trajectory_slice, i].tolist() + [last_step_reward[i]])
                # Compute GAE-Lambda advantage estimation (compute advantages using the values in the buffer taken from the model)
                deltas: numpy.ndarray = rewards[:-1] + self._discount_factor * values[1:] - values[:-1]
                advantages += self._discount_cumulative_sum(deltas, self._discount_factor * self._lambda_parameter).tolist()
                # Compute rewards-to-go
                rewards_to_go += (self._discount_cumulative_sum(rewards, self._discount_factor)[:-1]).tolist()
                # Get the observations, actions and log likelihoods of the trajectory
                observations_trajectory: numpy.ndarray = observations_array[trajectory_slice]
                actions_trajectory: numpy.ndarray = actions_array[trajectory_slice]
                log_likelihoods_trajectory: numpy.ndarray = log_likelihoods_array[trajectory_slice]
                lstm_states_trajectory: numpy.ndarray = lstm_states_array[trajectory_slice]
                # Serialize all observations, actions and log likelihoods for all parallel episodes
                serialized_observations += observations_trajectory[:, i].tolist()
                serialized_actions += actions_trajectory[:, i].tolist()
                serialized_log_likelihoods += log_likelihoods_trajectory[:, i].tolist()
                serialized_lstm_states += lstm_states_trajectory[:, i].tolist()
            trajectory_start_pointer = pointer.copy()
        # Get the numpy arrays out of all serialized data (plus advantages and rewards to go, already serialized)
        observations_array = numpy.array(serialized_observations)
        actions_array = numpy.array(serialized_actions)
        log_likelihoods_array = numpy.array(serialized_log_likelihoods)
        lstm_states_array = numpy.array(serialized_lstm_states)
        rewards_to_go_array: numpy.ndarray = numpy.array(rewards_to_go)
        advantages_array: numpy.ndarray = numpy.array(advantages)
        # Execute the advantage normalization trick
        # Note: make sure mean and std are not zero!
        advantage_mean: float = float(numpy.mean(advantages_array)) + 1e-8
        global_sum_squared: float = float(numpy.sum((advantages_array - advantage_mean) ** 2)) + 1e-8
        advantage_std: float = numpy.sqrt(global_sum_squared / advantages_array.size) + 1e-8
        # Adjust advantages according to the trick
        advantages_array = ((advantages_array - advantage_mean) / advantage_std)
        # Reset the buffer
        self._observations = []
        self._actions = []
        self._rewards = []
        self._values = []
        self._log_likelihoods = []
        self._last_step_rewards = []
        self._lstm_states = []
        # Reset pointer
        self._pointer: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=int)
        self._end_trajectories_pointers: [] = []
        # Return all the buffer components
        return [observations_array, actions_array, advantages_array, rewards_to_go_array, log_likelihoods_array, lstm_states_array]

    def finish_trajectory(self,
                          last_step_reward: numpy.ndarray):
        """
        Finish the trajectory storing all pointers data related to each parallel episode and resetting flags.

        :param last_step_reward: the last reward given by the environment or the last predicted value if last state is not terminal
        """
        # Add the last step rewards
        self._last_step_rewards.append(last_step_reward)
        # Save the current pointer as a trajectory end pointer
        self._end_trajectories_pointers.append(self._pointer.copy())
        # Since the episode is done reset flags for previous step
        self._episode_done_previous_step: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=bool)

    @property
    def size(self) -> int:
        """
        The size of the buffer at the current time (it is dynamic).

        :return: the integer size of the buffer
        """
        return int(numpy.sum(self._pointer))

    @staticmethod
    def _discount_cumulative_sum(vector: numpy.ndarray, discount: float) -> numpy.ndarray:
        """
        Compute discounted cumulative sums of vectors.
        Credits to rllab.

        :param vector: the vector on which to compute cumulative discounted sum (e.g. [x0, x1, x2])
        :return the discounted cumulative sum (e.g. [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x3])
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], vector[::-1], axis=0)[::-1]


class ProximalPolicyOptimizationRecurrent(Model):
    """
    Specific Proximal Policy Optimization powered by an LSTM network. It is possible to specify the number of LSTM
    layers and their forget bias. Everything else is the same as the default Proximal Policy Optimization algorithm.
    It is possible to use the config to stack any kind of layer on the LSTM.

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - discrete
        - continuous
    """

    def __init__(self,
                 name: str,
                 actor_hidden_layers_config: Config, critic_hidden_layers_config: Config,
                 lstm_layers_number: int = 1,
                 lstm_forget_bias: float = 1.0,
                 discount_factor: float = 0.99,
                 learning_rate_policy: float = 3e-4, learning_rate_value: float = 1e-3,
                 value_update_epochs: int = 80, policy_update_epochs: int = 80,
                 minibatch_size: int = 32,
                 lambda_parameter: float = 0.97,
                 clip_ratio: float = 0.2,
                 target_kl_divergence: float = 1e-2):
        # Define model attributes
        self.learning_rate_policy: float = learning_rate_policy
        self.learning_rate_value: float = learning_rate_value
        self.discount_factor: float = discount_factor
        # Define internal model attributes
        self._actor_hidden_layers_config: Config = actor_hidden_layers_config
        self._critic_hidden_layers_config: Config = critic_hidden_layers_config
        self._lstm_layers_number: int = lstm_layers_number
        self._lstm_forget_bias: float = lstm_forget_bias
        self._value_update_epochs: int = value_update_epochs
        self._policy_update_epochs: int = policy_update_epochs
        self._minibatch_size: int = minibatch_size
        self._lambda_parameter: float = lambda_parameter
        self._clip_ratio: float = clip_ratio
        self._target_kl_divergence: float = target_kl_divergence
        # Define proximal policy optimization empty attributes
        self.buffer: Buffer or None = None
        # Define internal proximal policy optimization empty attributes
        self._observations = None
        self._actions_predicted = None
        self._actions = None
        self._advantages = None
        self._rewards = None
        self._mask = None
        self._logits = None
        self._masked_logits = None
        self._lower_bound = None
        self._upper_bound = None
        self._expected_value = None
        self._std = None
        self._log_std = None
        self._log_likelihood_actions = None
        self._log_likelihood_predictions = None
        self._previous_log_likelihoods = None
        self._value_predicted = None
        self._ratio = None
        self._min_advantage = None
        self._value_stream_loss = None
        self._policy_stream_loss = None
        self._value_stream_optimizer = None
        self._policy_stream_optimizer = None
        self._approximated_kl_divergence = None
        self._approximated_entropy = None
        self._clip_fraction = None
        self._lstm_initial_states = None
        self._lstm_initial_states_layers = None
        self._lstm_initial_states_tuple = None
        self._lstm_cells = None
        self._lstm_states = None
        # Generate the base model
        super(ProximalPolicyOptimizationRecurrent, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.discrete)
        self._supported_action_space_types.append(SpaceType.continuous)

    def _define_graph(self):
        # Set the GAE buffer for the proximal policy optimization algorithm
        self.buffer: Buffer = Buffer(self._parallel, self.discount_factor, self._lambda_parameter)
        # Define the tensorflow model
        full_scope: str = self._scope + "/" + self._name
        with tensorflow.variable_scope(full_scope):
            # Define observations placeholder as an adaptable vector with shape Nx(O) where N is the number of examples and (O) the shape of the observation space
            # Note: it is the input of the model
            self._observations = tensorflow.placeholder(shape=(None, *self._observation_space_shape), dtype=tensorflow.float32, name="observations")
            # Define the actions placeholder as an adaptable vector with shape Nx(A) where N is the number of examples and (A) the shape of the action space
            self._actions = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="actions")
            # Define the rewards placeholder as an adaptable vector of floats (they are actually rewards-to-go computed in the buffer)
            # Note: the model gets the rewards from the buffer
            self._rewards = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="rewards")
            # Define the previous log-likelihood placeholder as an adaptable vector of float (previous because previously predicted by the model itself)
            # Note: the model gets the previous log-likelihoods from the buffer
            self._previous_log_likelihoods = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="previous_log_likelihood")
            # Define the advantages placeholder as an adaptable vector of floats (computed with GAE in the buffer)
            # Note: the model gets the advantages from the buffer once computed using GAE on the values
            self._advantages = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="advantages")
            # Define LSTM initial state placeholder as a vector with shape NxLx2x(O) where L is the number of LSTM layers, 2 is input and state, N is the number of examples and (O) the shape of the observation space
            self._lstm_initial_states = tensorflow.placeholder(shape=(None, self._lstm_layers_number, 2, *self._observation_space_shape), dtype=tensorflow.float32, name="lstm_initial_states")
            # Define a list of LSTM states divided by layers of shape Nx2x(O)
            self._lstm_initial_states_layers = tensorflow.unstack(self._lstm_initial_states, axis=1)
            # Generate a tuple as LSTM state divided by layers with content (initial_state_cell, initial_state_hidden)x(O)
            self._lstm_initial_states_tuple = tuple([tensorflow.contrib.rnn.LSTMStateTuple(tensorflow.transpose(self._lstm_initial_states_layers[layer_index], [1, 0, 2])[0], tensorflow.transpose(self._lstm_initial_states_layers[layer_index], [1, 0, 2])[1]) for layer_index in range(self._lstm_layers_number)])
            # Define the LSTM network
            self._lstm_layers = [tensorflow.contrib.rnn.LSTMCell(*self._observation_space_shape, forget_bias=self._lstm_forget_bias) for _ in range(self._lstm_layers_number)]
            self._lstm_cells = tensorflow.contrib.rnn.MultiRNNCell(self._lstm_layers, state_is_tuple=True)
            # Define the input of the LSTM as observations specifically shaped (with only 1 time step)
            lstm_input = tensorflow.reshape(self._observations, [tensorflow.shape(self._observations)[0], 1, self._observations.shape[1].value])
            # Get the output and the last state of the RNN
            lstm_outputs, self._lstm_states = tensorflow.nn.dynamic_rnn(self._lstm_cells, lstm_input, dtype=tensorflow.float32, initial_state=self._lstm_initial_states_tuple)
            # Define the policy stream
            # Note: this define the actor part of the model and its proper output (the predicted actions)
            with tensorflow.variable_scope("policy_stream"):
                # Define the policy stream network hidden layers from the config on top of the shared LSTM network (just the last prediction)
                policy_stream_hidden_layers_output = self._actor_hidden_layers_config.apply_hidden_layers(lstm_outputs[:, -1, :])
                # Change the model definition according to its action space type
                if self._agent_action_space_type == SpaceType.discrete:
                    # Define the mask placeholder
                    self._mask = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="mask")
                    # Define the logits as outputs with shape NxA where N is the size of the batch, A is the action size when its type is discrete
                    self._logits = tensorflow.layers.dense(policy_stream_hidden_layers_output, *self._agent_action_space_shape, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="logits")
                    # Compute the masked logits using the given additive mask
                    self._masked_logits = tensorflow.add(self._logits, self._mask)
                    # Define the predicted actions on the first shape dimension as a squeeze on the samples drawn from a categorical distribution over the logits
                    self._actions_predicted = tensorflow.squeeze(tensorflow.multinomial(logits=self._masked_logits, num_samples=1), axis=1)
                    # Define the log likelihood according to the categorical distribution on actions given and actions predicted
                    self._log_likelihood_actions, _ = self.get_categorical_log_likelihood(self._actions, self._logits, name="log_likelihood_actions")
                    self._log_likelihood_predictions, _ = self.get_categorical_log_likelihood(tensorflow.one_hot(self._actions_predicted, depth=self._agent_action_space_shape[0]), self._logits, name="log_likelihood_predictions")
                else:
                    # Define the boundaries placeholders to clip the actions
                    self._lower_bound = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="lower_bound")
                    self._upper_bound = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="upper_bound")
                    # Define the expected value as the output of the deep neural network with shape Nx(A) where N is the number of inputs, (A) is the action shape when its type is continuous
                    self._expected_value = tensorflow.layers.dense(policy_stream_hidden_layers_output, *self._agent_action_space_shape, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="expected_value")
                    # Define the log standard deviation and the standard deviation itself
                    self._log_std = tensorflow.get_variable(name="log_std", initializer=-0.5*numpy.ones(*self._agent_action_space_shape, dtype=numpy.float32))
                    self._std = tensorflow.exp(self._log_std, name="std")
                    # Define actions as the expected value summed up with a random gaussian vector multiplied by the standard deviation, clipped appropriately
                    self._actions_predicted = tensorflow.clip_by_value(self._expected_value + tensorflow.random_normal(tensorflow.shape(self._expected_value)) * self._std, self._lower_bound, self._upper_bound)
                    # Define the log likelihood according to the gaussian distribution on actions given and actions predicted
                    self._log_likelihood_actions = self.get_gaussian_log_likelihood(self._actions, self._expected_value, self._log_std, name="log_likelihood_actions")
                    self._log_likelihood_predictions = self.get_gaussian_log_likelihood(self._actions_predicted, self._expected_value, self._log_std, name="log_likelihood_predictions")
                # Define the ratio between the current log-likelihood and the previous one (when using exponential, minus is a division)
                self._ratio = tensorflow.exp(self._log_likelihood_actions - self._previous_log_likelihoods)
                # Define the minimum advantage with respect to clip ratio
                self._min_advantage = tensorflow.where(self._advantages > 0, (1 + self._clip_ratio) * self._advantages, (1 - self._clip_ratio) * self._advantages)
                # Define the policy stream loss as the mean of minimum between the advantages multiplied the ratio and the minimum advantage
                self._policy_stream_loss = -tensorflow.reduce_mean(tensorflow.minimum(self._ratio * self._advantages, self._min_advantage), name="policy_stream_loss")
                # Define the optimizer for the policy stream
                self._policy_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_policy).minimize(self._policy_stream_loss)
            # Define the value stream
            # Note: this is the critic part of the model and the system making the reward differentiable (the value computation)
            with tensorflow.variable_scope("value_stream"):
                # Define the value stream network hidden layers from the config and its output (a single float value) on top of the shared LSTM network (just the last prediction)
                value_stream_hidden_layers_output = self._critic_hidden_layers_config.apply_hidden_layers(lstm_outputs[:, -1, :])
                value_stream_output = tensorflow.layers.dense(value_stream_hidden_layers_output, 1, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="value")
                # Define value by squeezing the output of the value stream
                self._value_predicted = tensorflow.squeeze(value_stream_output, axis=1, name="value_predicted")
                # Define value stream loss as the mean squared error of the difference between rewards-to-go given and the predicted value
                self._value_stream_loss = tensorflow.reduce_mean((self._rewards - self._value_predicted) ** 2, name="value_stream_loss")
                # Define the optimizer for the value stream
                self._value_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_value).minimize(self._value_stream_loss)
            # Define approximated KL divergence (also used to early stop), approximated entropy and clip fraction for the summary
            self._approximated_kl_divergence = tensorflow.reduce_mean(self._previous_log_likelihoods - self._log_likelihood_actions, name="approximated_kl_divergence")
            self._approximated_entropy = tensorflow.reduce_mean(-self._log_likelihood_actions, name="approximated_entropy")
            self._clip_fraction = tensorflow.reduce_mean(tensorflow.cast(tensorflow.logical_or(self._ratio > (1 + self._clip_ratio), self._ratio < (1 - self._clip_ratio)), tensorflow.float32), name="clip_fraction")
            # Define the initializer
            self._initializer = tensorflow.variables_initializer(tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES, full_scope), name="initializer")

    def sample_action(self,
                      session,
                      observation_current: numpy.ndarray,
                      lstm_initial_state: [],
                      possible_actions: []):
        """
        Get the action sampled from the probability distribution of the model given the current observation and an optional mask.
        Also the LSTM internal state is returned.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param lstm_initial_state: the initial state of the LSTM required to take a decision
        :param possible_actions: the optional list used to remove certain actions from the prediction (with discrete action space) or to bound the actions predicted (with continuous action space)
        :return: the action predicted by the model, with the value estimated at the current state, the relative log-likelihood of the sampled action and the LSTM state after the action is taken
        """
        # Generate a one-hot encoded version of the observation if observation space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observation_current.reshape(-1)]
        # Compute with respect of the action space shape
        if self._agent_action_space_type == SpaceType.discrete:
            # If there is no possible actions list and the action space type is discrete generate a full pass-through mask otherwise generate a mask upon it
            if possible_actions is None:
                mask: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                mask: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                for i in range(self._parallel):
                    mask[i, possible_actions[i]] = 0.0
            # Get action, value and log-likelihood with discrete action space shape
            action, value, log_likelihood, lstm_state = session.run([self._actions_predicted, self._value_predicted, self._log_likelihood_predictions, self._lstm_states],
                                                                    feed_dict={
                                                                        self._observations: observation_current,
                                                                        self._mask: mask,
                                                                        self._lstm_initial_states: lstm_initial_state
                                                                    })
        else:
            # If there is no possible action list and the action space type is continuous generate an unbounded range, otherwise use the given range
            if possible_actions is None:
                lower_bound: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                upper_bound: numpy.ndarray = math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                lower_bound: numpy.ndarray = numpy.array(possible_actions)[:, 0]
                upper_bound: numpy.ndarray = numpy.array(possible_actions)[:, 1]
            # Get action, value and log-likelihood with continuous action space shape
            action, value, log_likelihood, lstm_state = session.run([self._actions_predicted, self._value_predicted, self._log_likelihood_predictions, self._lstm_states],
                                                                    feed_dict={
                                                                        self._observations: observation_current,
                                                                        self._lower_bound: lower_bound,
                                                                        self._upper_bound: upper_bound,
                                                                        self._lstm_initial_states: lstm_initial_state
                                                                    })
        # Return the predicted action, the estimated value, the log-likelihood and the LSTM state
        return action, value, log_likelihood, lstm_state

    def get_value_and_log_likelihood(self,
                                     session,
                                     action: numpy.ndarray,
                                     observation_current: numpy.ndarray,
                                     lstm_initial_state: [],
                                     possible_actions: []):
        """
        Get the estimated value of the given current observation and the log-likelihood of the given action.
        Also the LSTM internal state is returned.

        :param session: the session of tensorflow currently running
        :param action: the action of which to compute the log-likelihood
        :param observation_current: the current observation of the agent in the environment to estimate the value
        :param lstm_initial_state: the initial state of the LSTM required to take a decision
        :param possible_actions: the optional list used to remove certain actions from the prediction (with discrete action space) or to bound the actions predicted (with continuous action space)
        :return: the value estimated at the current state and the log-likelihood of the given action
        """
        # Generate a one-hot encoded version of the observation if observation space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observation_current.reshape(-1)]
        # Compute with respect of the action space shape
        if self._agent_action_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the action if action space type is discrete
            action: numpy.ndarray = numpy.eye(*self._agent_action_space_shape)[action.reshape(-1)]
            # If there is no possible actions list and the action space type is discrete generate a full pass-through mask otherwise generate a mask upon it
            if possible_actions is None:
                mask: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                mask: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                for i in range(self._parallel):
                    mask[i, possible_actions[i]] = 0.0
            # Get value and log-likelihood with discrete action space shape and LSTM state
            value, log_likelihood, lstm_state = session.run([self._value_predicted, self._log_likelihood_actions, self._lstm_states],
                                                            feed_dict={
                                                                self._observations: observation_current,
                                                                self._mask: mask,
                                                                self._actions: action,
                                                                self._lstm_initial_states: lstm_initial_state
                                                            })
        else:
            # If there is no possible action list and the action space type is continuous generate an unbounded range, otherwise use the given range
            if possible_actions is None:
                lower_bound: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                upper_bound: numpy.ndarray = math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                lower_bound: numpy.ndarray = possible_actions[:, 0]
                upper_bound: numpy.ndarray = possible_actions[:, 1]
            # Get value and log-likelihood with continuous action space shape and LSTM state
            value, log_likelihood, lstm_state = session.run([self._value_predicted, self._log_likelihood_actions, self._lstm_states],
                                                            feed_dict={
                                                                self._observations: observation_current,
                                                                self._lower_bound: lower_bound,
                                                                self._upper_bound: upper_bound,
                                                                self._actions: action,
                                                                self._lstm_initial_states: lstm_initial_state
                                                            })
        # Return the estimated value of the given current state, the log-likelihood of the given action and the LSTM state
        return value, log_likelihood, lstm_state

    def get_value(self,
                  session,
                  observation_current: numpy.ndarray,
                  lstm_initial_state: []):
        """
        Get the estimated value of the given current observation.
        Also the LSTM internal state is returned.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to estimate the value
        :param lstm_initial_state: the initial state of the LSTM required to take a decision
        :return: the value estimated at the current state
        """
        if self._observation_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the observation if observation space type is discrete
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observation_current.reshape(-1)]
        # Get the value predicted by the network at the current state and the LSTM state
        value, lstm_state = session.run([self._value_predicted, self._lstm_states],
                                        feed_dict={
                                            self._observations: observation_current,
                                            self._lstm_initial_states: lstm_initial_state
                                        })
        # Return the estimated value and the LSTM state
        return value, lstm_state

    def get_log_likelihood(self,
                           session,
                           action: numpy.ndarray,
                           observation_current: numpy.ndarray,
                           lstm_initial_state: [],
                           possible_actions: [] = None):
        """
        Get the the log-likelihood of the given action.
        Also the LSTM internal state is returned.

        :param session: the session of tensorflow currently running
        :param action: the action of which to compute the log-likelihood
        :param observation_current: the current observation of the agent in the environment
        :param lstm_initial_state: the initial state of the LSTM required to take a decision
        :param possible_actions: the optional list used to remove certain actions from the prediction (with discrete action space) or to bound the actions predicted (with continuous action space)
        :return: the log-likelihood of the given action
        """
        # Generate a one-hot encoded version of the observation if observation space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observation_current.reshape(-1)]
        # Compute with respect of the action space shape
        if self._agent_action_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the action if action space type is discrete
            action: numpy.ndarray = numpy.eye(*self._agent_action_space_shape)[action.reshape(-1)]
            # If there is no possible actions list and the action space type is discrete generate a full pass-through mask otherwise generate a mask upon it
            if possible_actions is None:
                mask: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                mask: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                for i in range(self._parallel):
                    mask[i, possible_actions[i]] = 0.0
            # Get log-likelihood with discrete action space shape and the LSTM state
            log_likelihood, lstm_state = session.run([self._log_likelihood_actions, self._lstm_states],
                                                     feed_dict={
                                                        self._observations: observation_current,
                                                        self._mask: mask,
                                                        self._actions: action,
                                                        self._lstm_initial_states: lstm_initial_state
                                                     })
        else:
            # If there is no possible action list and the action space type is continuous generate an unbounded range, otherwise use the given range
            if possible_actions is None:
                lower_bound: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                upper_bound: numpy.ndarray = math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                lower_bound: numpy.ndarray = possible_actions[:, 0]
                upper_bound: numpy.ndarray = possible_actions[:, 1]
            # Get log-likelihood with continuous action space shape and the LSTM state
            log_likelihood, lstm_state = session.run([self._log_likelihood_actions, self._lstm_states],
                                                     feed_dict={
                                                        self._observations: observation_current,
                                                        self._lower_bound: lower_bound,
                                                        self._upper_bound: upper_bound,
                                                        self._actions: action,
                                                        self._lstm_initial_states: lstm_initial_state
                                                     })
        # Return the log-likelihood of the given action and the LSTM state
        return log_likelihood, lstm_state

    def get_action_probabilities(self,
                                 session,
                                 observation_current: numpy.ndarray,
                                 lstm_initial_state: [],
                                 possible_actions: []) -> []:
        """
        Get all the action probabilities (softmax over masked logits if discrete, expected value and standard deviation if continuous) for the
        given current observation and an optional mask.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param lstm_initial_state: the initial state of the LSTM required to take a decision
        :param possible_actions: the optional list used to remove certain actions from the prediction (with discrete action space) or to bound the actions predicted (with continuous action space)
        :return: the list of action probabilities (softmax over masked logits or expected values and std wrapped in a list depending on the agent action space type)
        """
        # Generate a one-hot encoded version of the observation if observation space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observation_current.reshape(-1)]
        # Get the logits or the expected value and std of the distribution to compute the action probabilities depending on the action space shape
        if self._agent_action_space_type == SpaceType.discrete:
            # If there is no possible actions list and the action space type is discrete generate a full pass-through mask otherwise generate a mask upon it
            if possible_actions is None:
                mask: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                mask: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                for i in range(self._parallel):
                    mask[i, possible_actions[i]] = 0.0
            # Get logits on the current observation
            logits = session.run(self._masked_logits,
                                 feed_dict={
                                     self._observations: observation_current,
                                     self._mask: mask,
                                     self._lstm_initial_states: lstm_initial_state
                                 })
            # Return the softmax over the logits (probabilities of all actions)
            return softmax(logits)
        else:
            # If there is no possible action list and the action space type is continuous generate an unbounded range, otherwise use the given range
            if possible_actions is None:
                lower_bound: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
                upper_bound: numpy.ndarray = math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
            else:
                lower_bound: numpy.ndarray = possible_actions[:, 0]
                upper_bound: numpy.ndarray = possible_actions[:, 1]
            # Get logits on the current observation
            expected_value, std = session.run([self._expected_value, self._std],
                                              feed_dict={
                                                  self._observations: observation_current,
                                                  self._lower_bound: lower_bound,
                                                  self._upper_bound: upper_bound,
                                                  self._lstm_initial_states: lstm_initial_state
                                              })
            # Return the expected value and the standard deviation wrapped in a list
            return [expected_value, std]

    def update(self,
               session,
               batch: []):
        """
        Update the model weights (thus training the model) of the policy and value stream, given a batch of samples.
        Update is done through minibatches. Multiple updates of the policy (actor) and of the value (critic) are
        performed.

        :param session: the session of tensorflow currently running
        :param batch: a batch of samples each one consisting in a tuple of observation, action, advantage, reward, previous log-likelihood and LSTM state
        :return: the policy stream average loss of all minibatches, the value stream loss average over all updates and minibatches, the average KL divergence, approximated entropy and clip ratio of the policy
        """
        # Unpack the batch to feed the minibatches
        observations, actions, advantages, rewards, previous_log_likelihoods, lstm_states = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        # Generate a one-hot encoded version of the observations if observation space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observations).reshape(-1)]
        # Generate a one-hot encoded version of the actions if action space type is discrete
        if self._agent_action_space_type == SpaceType.discrete:
            actions: numpy.ndarray = numpy.eye(*self._agent_action_space_shape)[numpy.array(actions).reshape(-1)]
        # Run the policy optimizer of the model in training mode for the required amount of steps, also compute policy stream loss, approximated kl divergence, approximated entropy and clip fraction
        policy_stream_loss_average: float = 0.0
        approximated_kl_divergence_average: float = 0.0
        approximated_entropy_average: float = 0.0
        clip_fraction_average: float = 0.0
        for _ in range(self._policy_update_epochs):
            policy_update_minibatch_iterations: int = 0
            policy_stream_loss_total: float = 0.0
            approximated_kl_divergence_total: float = 0.0
            approximated_entropy_total: float = 0.0
            clip_fraction_total: float = 0.0
            for minibatch in self._get_minibatch(observations, actions, advantages, rewards, previous_log_likelihoods, lstm_states, self._minibatch_size):
                # Unpack the minibatch
                minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards, minibatch_previous_log_likelihoods, minibatch_lstm_states = minibatch
                # Update the policy
                _, policy_stream_loss, approximated_kl_divergence, approximated_entropy, clip_fraction = session.run([self._policy_stream_optimizer, self._policy_stream_loss, self._approximated_kl_divergence, self._approximated_entropy, self._clip_fraction],
                                                                                                                     feed_dict={
                                                                                                                         self._observations: minibatch_observations,
                                                                                                                         self._actions: minibatch_actions,
                                                                                                                         self._advantages: minibatch_advantages,
                                                                                                                         self._previous_log_likelihoods: minibatch_previous_log_likelihoods,
                                                                                                                         self._lstm_initial_states: minibatch_lstm_states
                                                                                                                     })
                policy_update_minibatch_iterations += 1
                policy_stream_loss_total += policy_stream_loss
                approximated_kl_divergence_total += approximated_kl_divergence
                approximated_entropy_total += approximated_entropy
                clip_fraction_total += clip_fraction
            # The average is only really saved on the last value update to know it at the end of all the update steps
            policy_stream_loss_average = policy_stream_loss_total / policy_update_minibatch_iterations
            approximated_kl_divergence_average = approximated_kl_divergence_total / policy_update_minibatch_iterations
            approximated_entropy_average = approximated_entropy_total / policy_update_minibatch_iterations
            clip_fraction_average = clip_fraction_total / policy_update_minibatch_iterations
            # If average approximated KL divergence over last policy update step (all minibatches) is above a certain threshold stop updating the policy for now (early stop)
            if approximated_kl_divergence_average > 1.5 * self._target_kl_divergence:
                break
        # Run the value optimizer of the model in training mode for the required amount of steps, also compute average value stream loss
        value_stream_loss_average: float = 0.0
        for _ in range(self._value_update_epochs):
            value_update_minibatch_iterations: int = 0
            value_stream_loss_total: float = 0.0
            for minibatch in self._get_minibatch(observations, actions, advantages, rewards, previous_log_likelihoods, lstm_states, self._minibatch_size):
                # Unpack the minibatch
                minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards, minibatch_previous_log_likelihoods, minibatch_lstm_states = minibatch
                # Update the value
                _, value_stream_loss = session.run([self._value_stream_optimizer, self._value_stream_loss],
                                                   feed_dict={
                                                        self._observations: minibatch_observations,
                                                        self._advantages: minibatch_advantages,
                                                        self._rewards: minibatch_rewards,
                                                        self._lstm_initial_states: minibatch_lstm_states
                                                    })
                value_update_minibatch_iterations += 1
                value_stream_loss_total += value_stream_loss
            # The average is only really saved on the last value update to know it at the end of all the update steps
            value_stream_loss_average = value_stream_loss_total / value_update_minibatch_iterations
        # Return all metrics
        return policy_stream_loss_average, value_stream_loss_average, approximated_kl_divergence_average, approximated_entropy_average, clip_fraction_average

    @property
    def warmup_steps(self) -> int:
        return 0

    @property
    def lstm_layers_number(self) -> int:
        """
        The number of LSTM layers powering the model.
        """
        return self._lstm_layers_number

    @staticmethod
    def _get_minibatch(observations: [], actions: [], advantages: [], rewards: [], previous_log_likelihoods: [], lstm_states: [],
                       minibatch_size: int) -> ():
        """
        Get a minibatch of the given minibatch size from the given batch (already unpacked).

        :param observations: the observations buffer in the batch
        :param actions: the actions buffer in the batch
        :param advantages: the advantages buffer in the batch
        :param rewards: the rewards buffer in the batch
        :param previous_log_likelihoods: the previous log-likelihoods buffer in the batch
        :param lstm_states: the LSTM states buffer in the batch
        :param minibatch_size: the size of the minibatch
        :return: a tuple minibatch of shuffled samples of the given size
        """
        # Get a list of random ids of the batch
        batch_random_ids = random.sample(range(len(observations)), len(observations))
        # Generate the minibatches by shuffling the batch
        minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards, minibatch_previous_log_likelihoods, minibatch_lstm_states = [], [], [], [], [], []
        for random_id in batch_random_ids:
            minibatch_observations.append(observations[random_id])
            minibatch_actions.append(actions[random_id])
            minibatch_advantages.append(advantages[random_id])
            minibatch_rewards.append(rewards[random_id])
            minibatch_previous_log_likelihoods.append(previous_log_likelihoods[random_id])
            minibatch_lstm_states.append(lstm_states[random_id])
            # Return the minibatch
            if len(minibatch_observations) % minibatch_size == 0:
                yield minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards, minibatch_previous_log_likelihoods, minibatch_lstm_states
                # Clear the minibatch
                minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards, minibatch_previous_log_likelihoods, minibatch_lstm_states = [], [], [], [], [], []

    @staticmethod
    def get_categorical_log_likelihood(actions_mask, logits, name: str):
        """
        Get log-likelihood for discrete action spaces (using a categorical distribution) on the given logits and with
        the action mask.
        It uses tensorflow and as such should only be called in the define method.

        :param actions_mask: the actions used to mask the log-likelihood on the logits
        :param logits: the logits of the neural network
        :param name: the name of the tensorflow operation
        :return: the log-likelihood according to categorical distribution
        """
        # Define the unmasked likelihood as the log-softmax of the logits
        log_likelihood_unmasked = tensorflow.nn.log_softmax(logits)
        # Return the categorical log-likelihood by summing over the first axis of the actions mask multiplied
        # by the log-likelihood on the logits (unmasked, this is the masking operation) and the unmasked likelihood
        return tensorflow.reduce_sum(actions_mask * log_likelihood_unmasked, axis=1, name=name), log_likelihood_unmasked

    @staticmethod
    def get_gaussian_log_likelihood(actions, expected_value, log_std, name: str):
        """
        Get log-likelihood for continuous action spaces (using a gaussian distribution) on the given expected value and
        log-std and with the actions.
        It uses tensorflow and as such should only be called in the define method.

        :param actions: the actions used to compute the log-likelihood tensor
        :param expected_value: the expected value of the gaussian distribution
        :param log_std: the log-std of the gaussian distribution
        :param name: the name of the tensorflow operation
        :return: the log-likelihood according to gaussian distribution
        """
        # Define the log-likelihood tensor for the gaussian distribution on the given actions
        log_likelihood_tensor = -0.5 * (((actions - expected_value) / (tensorflow.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + numpy.log(2 * numpy.pi))
        # Return the gaussian log-likelihood by summing over all the elements in the log-likelihood tensor defined above
        return tensorflow.reduce_sum(log_likelihood_tensor, axis=1, name=name)
