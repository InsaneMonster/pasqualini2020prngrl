# Import packages

import os

# Import usienarl

from usienarl.utils import command_line_parse

# Import required src

from rng_discrete_10_ppo import run as ppo_10_run
from rng_discrete_25_ppo import run as ppo_25_run
from rng_discrete_50_ppo import run as ppo_50_run
from rng_discrete_10_random import run as random_10_run
from rng_discrete_25_random import run as random_25_run
from rng_discrete_50_random import run as random_50_run

if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run all RNG Discrete experiments
    # random_10_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    # random_25_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    # random_50_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ppo_10_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ppo_25_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ppo_50_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
