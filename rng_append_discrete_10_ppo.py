# Import packages

import tensorflow
import logging
import os

# Import usienarl

from usienarl import Config, LayerType
from usienarl.utils import run_experiment, command_line_parse

# Import required src

from src.rng_append_discrete_environment import RNGAppendDiscreteEnvironment
from src.rng_append_experiment import RNGAppendExperiment
from src.proximal_policy_optimization_recurrent import ProximalPolicyOptimizationRecurrent
from src.ppo_agent_recurrent import PPOAgentRecurrent

# Define utility functions to run the experiment


def _define_ppo_model(actor_config: Config, critic_config: Config) -> ProximalPolicyOptimizationRecurrent:
    # Define attributes
    lstm_layers_number: int = 2
    lstm_forget_bias: float = 1.0
    learning_rate_policy: float = 3e-4
    learning_rate_advantage: float = 1e-3
    discount_factor: float = 1.0
    value_steps_per_update: int = 80
    policy_steps_per_update: int = 80
    minibatch_size: int = 32
    lambda_parameter: float = 0.95
    clip_ratio: float = 0.2
    target_kl_divergence: float = 0.01
    # Return the model
    return ProximalPolicyOptimizationRecurrent("ppo_model", actor_config, critic_config,
                                               lstm_layers_number, lstm_forget_bias,
                                               discount_factor,
                                               learning_rate_policy, learning_rate_advantage,
                                               value_steps_per_update, policy_steps_per_update,
                                               minibatch_size,
                                               lambda_parameter,
                                               clip_ratio,
                                               target_kl_divergence)


def _define_agent(model: ProximalPolicyOptimizationRecurrent) -> PPOAgentRecurrent:
    # Define attributes
    update_every_episodes: int = 1000
    # Return the agent
    return PPOAgentRecurrent("ppo_agent", model, update_every_episodes)


def run(workspace: str,
        iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Generate RNG Append Discrete Environments with varying lambda parameters
    append_sequence_size: int = 10
    max_moves: int = 100
    environment_l1: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_10_env_l1", append_sequence_size, max_moves,
                                                                                lambda_parameter=1.0)
    # environment_l10: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_10_env_l10", append_sequence_size, max_moves,
    #                                                                              lambda_parameter=10.0)
    # environment_l25: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_10_env_l25", append_sequence_size, max_moves,
    #                                                                              lambda_parameter=25.0)
    # environment_l50: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_10_env_l50", append_sequence_size, max_moves,
    #                                                                              lambda_parameter=50.0)
    # environment_l100: RNGAppendDiscreteEnvironment = RNGAppendDiscreteEnvironment("rngad_10_env_l100", append_sequence_size, max_moves,
    #                                                                               lambda_parameter=100.0)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [256, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_1")
    nn_config.add_hidden_layer(LayerType.dense, [128, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_2")
    nn_config.add_hidden_layer(LayerType.dense, [64, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_3")
    # Define model
    inner_model: ProximalPolicyOptimizationRecurrent = _define_ppo_model(actor_config=nn_config, critic_config=nn_config)
    # Define agent
    ppo_agent: PPOAgentRecurrent = _define_agent(inner_model)
    # Define experiments (validation and test thresholds vary with lambda parameters)
    validation_threshold: float = 0.7
    test_threshold: float = 0.7
    experiment_baseline_l1: RNGAppendExperiment = RNGAppendExperiment("rngad_10_exp_baseline_l1",
                                                                      validation_threshold=validation_threshold,
                                                                      test_threshold=test_threshold,
                                                                      environment=environment_l1, agent=ppo_agent)
    # experiment_baseline_l10: RNGAppendExperiment = RNGAppendExperiment("rngad_10_exp_baseline_l10",
    #                                                                    validation_threshold=validation_threshold * 10.0,
    #                                                                    test_threshold=test_threshold * 10.0,
    #                                                                    environment=environment_l10, agent=ppo_agent)
    # experiment_baseline_l25: RNGAppendExperiment = RNGAppendExperiment("rngad_10_exp_baseline_l25",
    #                                                                    validation_threshold=validation_threshold * 25.0,
    #                                                                    test_threshold=test_threshold * 25.0,
    #                                                                    environment=environment_l25, agent=ppo_agent)
    # experiment_baseline_l50: RNGAppendExperiment = RNGAppendExperiment("rngad_10_exp_baseline_l50",
    #                                                                    validation_threshold=validation_threshold * 50.0,
    #                                                                    test_threshold=test_threshold * 50.0,
    #                                                                    environment=environment_l50, agent=ppo_agent)
    # experiment_baseline_l100: RNGAppendExperiment = RNGAppendExperiment("rngad_10_exp_baseline_l100",
    #                                                                     validation_threshold=validation_threshold * 100.0,
    #                                                                     test_threshold=test_threshold * 100.0,
    #                                                                     environment=environment_l100, agent=ppo_agent)
    # Define experiment data
    saves_to_keep: int = 5
    plots_dpi: int = 150
    parallel: int = 100
    training_episodes: int = 2000
    validation_episodes: int = 100
    training_validation_volleys: int = 35
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
