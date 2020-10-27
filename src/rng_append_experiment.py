# Import packages

import logging

# Import usienarl

from usienarl import Experiment, Agent, Interface

# Import required src

from src.rng_append_discrete_environment import RNGAppendDiscreteEnvironment


class RNGAppendExperiment(Experiment):
    """
    RNG Append task experiment.
    Note: it is compatible both with a discrete and a continuous environment.

    It uses a validation threshold and a test threshold.
    """

    def __init__(self,
                 name: str,
                 validation_threshold: float,
                 test_threshold: float,
                 environment: RNGAppendDiscreteEnvironment,
                 agent: Agent,
                 interface: Interface = None):
        # Generate the base experiment
        super(RNGAppendExperiment, self).__init__(name, environment, agent, interface)
        # Define internal attributes
        self._validation_threshold: float = validation_threshold
        self._test_threshold: float = test_threshold

    def _is_validated(self,
                      logger: logging.Logger) -> bool:
        # Check if average validation total reward on the last validation volley is greater than validation threshold
        if self.validation_volley.avg_total_reward >= self._validation_threshold:
            return True
        return False

    def _is_successful(self,
                       logger: logging.Logger) -> bool:
        # Check if average test total reward over all test volleys is greater than test threshold
        if self.avg_test_avg_total_reward >= self._test_threshold:
            return True
        return False
