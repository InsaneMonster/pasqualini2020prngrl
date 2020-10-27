# Import packages

import os

# Import usienarl

from usienarl.utils import command_line_parse

# Import required src

from rng_append_discrete_2_ppo import run as ppo_2_run
from rng_append_discrete_5_ppo import run as ppo_5_run
from rng_append_discrete_10_ppo import run as ppo_10_run
from rng_append_discrete_2_random import run as random_2_run
from rng_append_discrete_5_random import run as random_5_run
from rng_append_discrete_10_random import run as random_10_run

if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run all RNG Append Discrete experiments
    random_2_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    random_5_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    random_10_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ppo_2_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ppo_5_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ppo_10_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
