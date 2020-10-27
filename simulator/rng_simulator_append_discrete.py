# Import packages

import numpy
import os
import datetime
import itertools

# Import nistrng

from nistrng import run_all_battery, check_eligibility_all_battery, SP800_22R1A_BATTERY


class RNGSimulatorAppendDiscrete:
    """
    RNG Simulator controlling one sequence of bits in by appending possible permutations of bits to it (discrete actions).
    """
    def __init__(self,
                 name: str,
                 append_sequence_size: int,
                 max_moves: int,
                 can_render: bool = True):
        # Make sure parameters are correct
        assert (append_sequence_size > 0 and max_moves > 0)
        # Define attributes
        self._name: str = name
        self._append_sequence_size: int = append_sequence_size
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
        # Generate all the possible sequences of bits to append
        self._append_sequences: [] = list(itertools.product(range(2), repeat=self._append_sequence_size))
        # Note: compute the eligible test battery over a set of zeros of the maximum length (the length at which the final score is computed)
        self._eligible_battery: dict = check_eligibility_all_battery(numpy.zeros(self._max_moves * self._append_sequence_size, dtype=float), SP800_22R1A_BATTERY)
        # Initialize the number of steps and episodes
        self._episode_number: int = -1
        self._step_number: int = 0
        # Define empty attributes
        self._current_state: numpy.ndarray or None = None
        self._previous_state: numpy.ndarray or None = None
        self._last_appended_sequence: numpy.ndarray or None = None
        self._last_received_action: int or None = None
        self._last_computed_score: float or None = None
        self._episode_done: bool or None = None

    def reset(self) -> numpy.ndarray:
        """
        Reset the simulator to a set of zeros (to leverage generation).

        :return: the start sequence wrapped in a numpy array
        """
        # Reset the step number and increment the episode number
        self._step_number = 0
        self._episode_number += 1
        self._episode_done = False
        # Assign a set of zeros as the starting sequence
        self._current_state = numpy.zeros(self._append_sequence_size, dtype=int)
        # Return the current state
        return self._current_state

    def step(self,
             action: int) -> ():
        """
        Executes a step in the simulator, append a sequence of bit defined by the given action.
        Note: the first appended sequence replaces the starting sequence of zeros, to leverage generation.

        :param action: the action is the id of a sequence of 0 and 1 to append to the state
        :return: the new increased sequence wrapped in a numpy array, the score at the current step and the episode done flag
        """
        # Make sure the action is feasible
        assert(0 <= action <= len(self._append_sequences) - 1)
        # Increment the step number
        self._step_number += 1
        # Save all data for renderers
        self._previous_state = numpy.copy(self._current_state)
        self._last_received_action = action
        # Update the state sequence of the state by appending the sequence of bits identified by the action
        self._last_appended_sequence = numpy.array(self._append_sequences[action])
        # If this is the first step do not append (we are interested in evaluating only the generation)
        if self._step_number == 1:
            self._current_state = numpy.copy(self._last_appended_sequence)
        else:
            self._current_state = numpy.append(self._current_state, self._last_appended_sequence)
        # Save the score only on termination
        self._last_computed_score = 0.0
        # Note: termination is (max_moves + 1) because the starting sequence of zeros is removed
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
    def append_sequences(self) -> []:
        """
        The list of sequences that is possible to append.
        """
        return self._append_sequences

    @property
    def append_sequence_size(self) -> int:
        """
        The size of each one of the sequences to append.
        """
        return self._append_sequence_size

    @property
    def append_sequences_number(self) -> int:
        """
        The number of the possible sequences to append.
        """
        return len(self._append_sequences)

    @property
    def current_state(self) -> numpy.ndarray:
        """
        The current sequence of bits in the simulator.
        """
        return self._current_state

    @property
    def last_appended_sequence(self) -> numpy.ndarray:
        """
        The last appended sequence of bits in the state.
        """
        return self._last_appended_sequence
