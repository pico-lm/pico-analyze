"""
This file contains the code for learning dynamics analysis.
"""

from src.utils.data import generate_learning_dynamics_data


repo_id = "pico-lm/demo"
branch = "demo-1"

"""
NOTE: 
    1. User specifies the repo_id and branch or the folder directly containing the run 
    2. The user specifies what sort of learning dynamics they want to analyze in terms of metrics
        and presumably also for what time steps they want to analyze the model.
    3a. If the user specifies a branch, we will download the data for all commits in that branch
    3b. If the user specifies a folder, we will download the data for all commits in that folder
    4. We will then compute the metrics for each commit and time step
    5. We will then save the metrics to a file and possibly plot the resulting data.

    --> Maybe we save that data directly to the wandb repo? 
"""


for step, data in generate_learning_dynamics_data(
    repo_id=repo_id,
    branch=branch
):
    print(step, data)
    break