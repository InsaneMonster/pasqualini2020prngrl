# Import packages

import numpy
import random
import os
import datetime

# Import nistrng

from nistrng import run_all_battery, check_eligibility_all_battery, SP800_22R1A_BATTERY


class RNGSimulatorDiscrete:
    """
    RNG Simulator Discrete controlling one sequence of bits and operating with discrete actions.
    """
    def __init__(self,
                 name: str,
                 sequence_size: int,
                 max_moves: int,
                 can_render: bool = True):
        # Make sure parameters are correct
        assert (sequence_size > 0 and max_moves > 0)
        # Define attributes
        self._name: str = name
        self._numeric_sequence_size: int = sequence_size
        self._max_moves: int = max_moves
        # Generate a directory to render at this time stamp (if can render)
        self._render_directory: str or None = None
        if can_render:
            self._render_directory: str = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "render"), datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
            if not os.path.isdir(self._render_directory):
                try:
                    os.makedirs(self._render_directory)
                except FileExistsError:
                    pass
        # Note: 8-bit packed sequences
        self._binary_sequence_size: int = sequence_size * 8
        # Note: two highly correlated sequences as default seed states
        self._seed_states: [] = [numpy.ones(self._binary_sequence_size, dtype=int), numpy.zeros(self._binary_sequence_size, dtype=int)]
        # Note: compute the eligible test battery over one of the seed states (only the length matters)
        self._eligible_battery: dict = check_eligibility_all_battery(self._seed_states[1], SP800_22R1A_BATTERY)
        # Initialize the number of steps and episodes
        self._episode_number: int = -1
        self._step_number: int = 0
        # Define empty attributes
        self._current_state: numpy.ndarray or None = None
        self._previous_state: numpy.ndarray or None = None
        self._last_received_action: int or None = None
        self._last_computed_score: float or None = None
        self._episode_done: bool or None = None

    def reset(self) -> numpy.ndarray:
        """
        Reset the simulator to an highly correlated start (seed) sequence.

        :return: the start sequence wrapped in a numpy array
        """
        # Reset the step number and increment the episode number
        self._step_number = 0
        self._episode_number += 1
        self._episode_done = False
        # Assign a random seed state to the current state
        self._current_state = numpy.copy(random.choice(self._seed_states))
        # Return the current state
        return self._current_state

    def step(self,
             action: int) -> ():
        """
        Executes a step in the simulator, setting one bit.

        :param action: the action defining the value of the bit (0 or 1) and the index at which to set it
        :return: the new sequence wrapped in a numpy array, the score at the current step and the episode done flag
        """
        # Make sure the action is feasible
        assert (0 <= action < self._binary_sequence_size * 2)
        # Increment the step number
        self._step_number += 1
        # Save all data for renderers
        self._previous_state = numpy.copy(self._current_state)
        self._last_received_action = action
        # Update the binary representation of the state
        if action % 2 == 1:
            self._current_state[action // 2] = 0
        else:
            self._current_state[action // 2] = 1
        # Save the score only on termination
        self._last_computed_score = 0.0
        if self._step_number >= self._max_moves:
            self._last_computed_score = self._compute_score()
            self._episode_done = True
        # Return the new state with the score computed using NIST
        return numpy.copy(self._current_state), self._last_computed_score, self._episode_done

    def render(self):
        """
        Render on file the current step of the simulator (state, action, new state and score if termination is reached).
        """
        # If there is no render directory stop here
        if self._render_directory is None or not self._render_directory:
            return
        render_file: str = os.path.join(self._render_directory, self._name) + "_ep_" + str(self._episode_number) + ".txt"
        with open(render_file, "a") as file:
            data_str: str = "----------------------------------------\n"
            data_str += "s" + str(self._step_number - 1) + ":\t" + str(self._previous_state) + "\n"
            data_str += "a" + str(self._step_number - 1) + ":\t" + str(self._last_received_action) + "\n"
            data_str += "s" + str(self._step_number) + ":\t" + str(self._current_state) + "\n"
            if self._episode_done:
                data_str += "scr" + str(self._step_number) + ":\t" + str(self._last_computed_score) + "\n"
            data_str += "----------------------------------------\n"
            file.write(data_str)

    def _compute_score(self) -> float:
        """
        Compute the score of the current sequence by running the NIST test battery.

        :return: the float score computed according to the test battery
        """
        # Execute the NIST test battery (only eligible tests) on the current sequence encoded in binary format
        results: [] = run_all_battery(self._current_state, self._eligible_battery, False)
        # Generate a score list from the results
        scores: [] = []
        for result, _ in results:
            # Make sure there is no computational error inside the test battery wreaking havoc inside the loss
            if not numpy.isnan(result.score):
                scores.append(int(result.passed) * result.score)
            else:
                scores.append(0.0)
        # Compute the reward averaging the elements in the score list
        return numpy.round(numpy.average(numpy.array(scores)), 2)

    @property
    def name(self) -> str:
        """
        The name of the simulator.
        """
        return self._name

    @property
    def numeric_sequence_size(self) -> int:
        """
        The size of the numeric sequence.
        """
        return self._numeric_sequence_size

    @property
    def binary_sequence_size(self) -> int:
        """
        The size of the binary sequence (8-bit packed numeric sequence).
        """
        return self._binary_sequence_size

    @property
    def current_state(self) -> numpy.ndarray:
        """
        The current sequence of bits in the simulator.
        """
        return self._current_state
