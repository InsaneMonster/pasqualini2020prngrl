# Import packages

import logging
import os

# Import usienarl

from usienarl.utils import run_experiment, command_line_parse
from usienarl.agents import RandomAgent

# Import required src

from src.rng_append_discrete_environment import RNGAppendDiscreteEnvironment
from src.rng_append_experiment import RNGAppendExperiment

# Define utility functions to run the experiment


def _define_agent() -> RandomAgent:
    # Return the agent
    return RandomAgent("random_agent")


def run(workspace: str,
        iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Generate RNG Append Discrete Environments with varying lambda parameters
    append_sequence_size: int = 5
    max_moves: int = 100
    environment_l1: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_5_env_l1", append_sequence_size, max_moves,
                                                                                lambda_parameter=1.0)
    # environment_l10: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_5_env_l10", append_sequence_size, max_moves,
    #                                                                              lambda_parameter=10.0)
    # environment_l25: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_5_env_l25", append_sequence_size, max_moves,
    #                                                                              lambda_parameter=25.0)
    # environment_l50: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_5_env_l50", append_sequence_size, max_moves,
    #                                                                              lambda_parameter=50.0)
    # environment_l100: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_5_env_l100", append_sequence_size, max_moves,
    #                                                                               lambda_parameter=100.0)
    # Define agent
    random_agent: RandomAgent = _define_agent()
    # Define experiments (validation and test thresholds vary with lambda parameters)
    validation_threshold: float = 0.7
    test_threshold: float = 0.7
    experiment_baseline_l1: RNGAppendExperiment = RNGAppendExperiment("rngad_5_exp_baseline_l1",
                                                                      validation_threshold=validation_threshold,
                                                                      test_threshold=test_threshold,
                                                                      environment=environment_l1, agent=random_agent)
    # experiment_baseline_l10: RNGAppendExperiment = RNGAppendExperiment("rngad_5_exp_baseline_l10",
    #                                                                    validation_threshold=validation_threshold * 10.0,
    #                                                                    test_threshold=test_threshold * 10.0,
    #                                                                    environment=environment_l10, agent=random_agent)
    # experiment_baseline_l25: RNGAppendExperiment = RNGAppendExperiment("rngad_5_exp_baseline_l25",
    #                                                                    validation_threshold=validation_threshold * 25.0,
    #                                                                    test_threshold=test_threshold * 25.0,
    #                                                                    environment=environment_l25, agent=random_agent)
    # experiment_baseline_l50: RNGAppendExperiment = RNGAppendExperiment("rngad_5_exp_baseline_l50",
    #                                                                    validation_threshold=validation_threshold * 50.0,
    #                                                                    test_threshold=test_threshold * 50.0,
    #                                                                    environment=environment_l50, agent=random_agent)
    # experiment_baseline_l100: RNGAppendExperiment = RNGAppendExperiment("rngad_5_exp_baseline_l100",
    #                                                                     validation_threshold=validation_threshold * 100.0,
    #                                                                     test_threshold=test_threshold * 100.0,
    #                                                                     environment=environment_l100, agent=random_agent)
    # Define experiment data
    saves_to_keep: int = 1
    plots_dpi: int = 150
    parallel: int = 10
    training_episodes: int = 100
    validation_episodes: int = 100
    training_validation_volleys: int = 1
    test_episodes: int = 100
    test_volleys: int = 10
    episode_length_max: int = 10000
    # Run experiments
    run_experiment(logger=logger, experiment=experiment_baseline_l1,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    # run_experiment(logger=logger, experiment=experiment_baseline_l10,
    #                file_name=__file__, workspace_path=workspace,
    #                training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
    #                training_validation_volleys=training_validation_volleys,
    #                test_volleys_episodes=test_episodes, test_volleys=test_volleys,
    #                episode_length=episode_length_max, parallel=parallel,
    #                render_during_training=render_training, render_during_validation=render_validation,
    #                render_during_test=render_test,
    #                iterations=iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    # run_experiment(logger=logger, experiment=experiment_baseline_l25,
    #                file_name=__file__, workspace_path=workspace,
    #                training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
    #                training_validation_volleys=training_validation_volleys,
    #                test_volleys_episodes=test_episodes, test_volleys=test_volleys,
    #                episode_length=episode_length_max, parallel=parallel,
    #                render_during_training=render_training, render_during_validation=render_validation,
    #                render_during_test=render_test,
    #                iterations=iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    # run_experiment(logger=logger, experiment=experiment_baseline_l50,
    #                file_name=__file__, workspace_path=workspace,
    #                training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
    #                training_validation_volleys=training_validation_volleys,
    #                test_volleys_episodes=test_episodes, test_volleys=test_volleys,
    #                episode_length=episode_length_max, parallel=parallel,
    #                render_during_training=render_training, render_during_validation=render_validation,
    #                render_during_test=render_test,
    #                iterations=iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    # run_experiment(logger=logger, experiment=experiment_baseline_l100,
    #                file_name=__file__, workspace_path=workspace,
    #                training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
    #                training_validation_volleys=training_validation_volleys,
    #                test_volleys_episodes=test_episodes, test_volleys=test_volleys,
    #                episode_length=episode_length_max, parallel=parallel,
    #                render_during_training=render_training, render_during_validation=render_validation,
    #                render_during_test=render_test,
    #                iterations=iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)


if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run this experiment
    run(workspace_path, experiment_iterations, render_during_training, render_during_validation, render_during_test)
