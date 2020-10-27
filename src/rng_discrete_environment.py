# Import packages

import logging
import numpy
import random

# Import usienarl

from usienarl.environment import Environment, SpaceType

# Import required src

from simulator.rng_simulator_discrete import RNGSimulatorDiscrete


class RNGDiscreteEnvironment(Environment):
    """
    Base environment class for the RNG Discrete task.
    """

    def __init__(self,
                 name: str,
                 sequence_size: int,
                 max_moves: int,
                 lambda_parameter: float = 100.0,
                 simulator_can_render: bool = True):
        # Make sure parameters are correct
        assert (sequence_size > 0 and max_moves > 0)
        # Define environment attributes
        self._sequence_size: int = sequence_size
        self._max_moves: int = max_moves
        self._lambda_parameter: float = lambda_parameter
        self._simulator_can_render: bool = simulator_can_render
        # Define environment empty attributes
        self._rng_simulators: [] = []
        self._last_step_episode_done_flags: numpy.ndarray or None = None
        self._last_step_states: numpy.ndarray or None = None
        # Generate the base environment
        super(RNGDiscreteEnvironment, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger) -> bool:
        # Close all previous environments, if any
        self.close(logger, None)
        # Generate all new parallel simulators
        for i in range(self._parallel):
            self._rng_simulators.append(RNGSimulatorDiscrete(self._name + "_" + str(i), self._sequence_size, self._max_moves, self._simulator_can_render))
        # Setup attributes
        self._last_step_episode_done_flags = numpy.zeros(self._parallel, dtype=bool)
        self._last_step_states: numpy.ndarray = numpy.zeros(self._parallel, dtype=int)
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        pass

    def close(self,
              logger: logging.Logger,
              session):
        # Clear all the simulators
        self._rng_simulators = []

    def reset(self,
              logger: logging.Logger,
              session) -> numpy.ndarray:
        # Prepare list of return values
        start_states: [] = []
        # Reset all parallel simulators
        self._last_step_episode_done_flags = numpy.zeros(self._parallel, dtype=bool)
        self._last_step_states = numpy.zeros((self._parallel, *self.state_space_shape), dtype=int)
        for i in range(len(self._rng_simulators)):
            start_state = self._rng_simulators[i].reset()
            start_states.append(start_state)
        # Return start states wrapped in a numpy array
        return numpy.array(start_states)

    def step(self,
             logger: logging.Logger,
             session,
             action: numpy.ndarray) -> ():
        # Make sure the action is properly sized
        assert (len(self._rng_simulators) == action.shape[0])
        # Prepare list of return values
        states: [] = []
        rewards: [] = []
        episode_done_flags: [] = []
        # Make a step in all non completed simulator
        for i in range(len(self._rng_simulators)):
            # Add dummy values to return if this parallel simulator episode is already done
            if self._last_step_episode_done_flags[i]:
                states.append(self._last_step_states[i])
                rewards.append(0.0)
                episode_done_flags.append(True)
                continue
            # Execute the step in this parallel simulator
            state_next, score, episode_done = self._rng_simulators[i].step(action[i])
            # Compute the reward from the score
            reward: float = score * self._lambda_parameter
            # Save results
            states.append(state_next)
            rewards.append(reward)
            episode_done_flags.append(episode_done)
            # Update last step flags and states
            self._last_step_episode_done_flags[i] = episode_done
            self._last_step_states[i] = state_next
        # Return the new states, rewards and episode done flags wrapped in numpy array
        return numpy.array(states), numpy.array(rewards), numpy.array(episode_done_flags)

    def render(self,
               logger: logging.Logger,
               session):
        # Make sure there is at least a parallel simulator
        assert (len(self._rng_simulators) > 0)
        # Render all the simulators
        for i in range(len(self._rng_simulators)):
            self._rng_simulators[i].render()

    def sample_action(self,
                      logger: logging.Logger,
                      session) -> numpy.ndarray:
        # Prepare list of return values
        actions: [] = []
        # Get all the possible actions
        possible_actions: [] = self.possible_actions(logger, session)
        # Get a random action for each parallel simulator
        for i in range(len(self._rng_simulators)):
            actions.append(random.choice(possible_actions[i]))
        # Return sampled actions wrapped in numpy array
        return numpy.array(actions)

    def possible_actions(self,
                         logger: logging.Logger,
                         session) -> []:
        # Prepare list of return values
        possible_actions: [] = []
        # Get the possible actions at the current state for each simulator
        for i in range(len(self._rng_simulators)):
            possible_actions.append([action for action in range(2 * self._rng_simulators[i].binary_sequence_size)])
        # Return the possible actions list
        return possible_actions

    @property
    def state_space_type(self) -> SpaceType:
        # Always continuous
        return SpaceType.continuous

    @property
    def state_space_shape(self) -> ():
        # Make sure there is at least a parallel environment
        assert (len(self._rng_simulators) > 0)
        # Get the number of available states (continuous, so size of the state) in the simulator
        return numpy.zeros(self._rng_simulators[0].binary_sequence_size).shape

    @property
    def action_space_type(self) -> SpaceType:
        # Always discrete
        return SpaceType.discrete

    @property
    def action_space_shape(self) -> ():
        # Make sure there is at least a parallel environment
        assert (len(self._rng_simulators) > 0)
        # Get the number of available actions in the simulator
        return 2 * self._rng_simulators[0].binary_sequence_size,

    @property
    def rng_simulators(self) -> []:
        """
        The list of RNG simulators running behind the environment.
        """
        return self._rng_simulators

    @property
    def max_moves(self) -> int:
        """
        The number of moves the environment executes before setting the episode done flag.
        """
        return self._max_moves
