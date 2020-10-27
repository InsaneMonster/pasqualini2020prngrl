# Import packages

import logging
import numpy

# Import usienarl

from usienarl import Interface, SpaceType

# Import required src

from src.rng_discrete_environment import RNGDiscreteEnvironment


class RNGDiscreteWanderInterface(Interface):
    """
    Wander interface for RNG Discrete environment.

    It forces the agent to always set the chosen bit to a value different from the current one.
    """

    def __init__(self,
                 environment: RNGDiscreteEnvironment):
        # Generate the base interface
        super(RNGDiscreteWanderInterface, self).__init__(environment)

    def agent_action_to_environment_action(self,
                                           logger: logging.Logger,
                                           session,
                                           agent_action: numpy.ndarray) -> numpy.ndarray:
        # Just return the agent action
        return agent_action

    def environment_action_to_agent_action(self,
                                           logger: logging.Logger,
                                           session,
                                           environment_action: numpy.ndarray) -> numpy.ndarray:
        # Just return the environment action
        return environment_action

    def environment_state_to_observation(self,
                                         logger: logging.Logger,
                                         session,
                                         environment_state: numpy.ndarray) -> numpy.ndarray:
        # Just return the environment state
        return environment_state

    def possible_agent_actions(self,
                               logger: logging.Logger,
                               session) -> []:
        # Prepare list of return values (discarding the ones coming from the environment)
        possible_actions: [] = []
        # Get the possible actions at the current state for each simulator in the environment
        # Note: with this interface it's not possible to change a bit with itself (i.e. stay in one place for more than one step)
        for i in range(len(self._environment.rng_simulators)):
            current_state: numpy.ndarray = self._environment.rng_simulators[i].current_state
            possible_actions_simulator: [] = []
            for bit_idx in range(current_state.size):
                if current_state[bit_idx] == 0:
                    possible_actions_simulator += [bit_idx * 2]
                else:
                    possible_actions_simulator += [bit_idx * 2 + 1]
            possible_actions.append(possible_actions_simulator)
        # Return the possible actions list
        return possible_actions

    @property
    def observation_space_type(self) -> SpaceType:
        # Just return the environment state space type
        return self._environment.state_space_type

    @property
    def observation_space_shape(self) -> ():
        # Just return the environment state space shape
        return self._environment.state_space_shape

    @property
    def agent_action_space_type(self) -> SpaceType:
        # Just return the environment action space type
        return self._environment.action_space_type

    @property
    def agent_action_space_shape(self) -> ():
        # Just return the environment action space shape
        return self._environment.action_space_shape
