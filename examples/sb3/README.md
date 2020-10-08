When running sb3_sac.py, run `pip3 install -r requirements_sb3.txt` in a separate virtual environment. A corresponding requirements_sb2.txt will be made, but note that for SB2, you will need python 3.7. 

The following are general descriptions of the files found here.

sb3_sac.py: This trains an agent using SAC from stable_baselines3 to interact with fishing-v1 i.e. the continous action space fishing environment. The hyperparameters here are not leading to consistent near-optimal performance just yet. I recently changed the episode length on the base environment, which degraded performance here even more.

tuning_sb3_sac.py: This is the script that executes a round of tuning.

tuning_sb3_utils.py: This script contains all the workings to carry out the tuning round that is called by tuning_sb3_sac.py. Of note here are the test distributions of hyperparameter values that you can adjust. This section is found at the bottom of the script.

callbacks_sb3.py: This also contains some objects that are used when tuning, but nothing that I have edited from rl-zoo. 

plot.py: This creates an .png of an averaged trajectory from an agent over 100 episodes. 

sb2_recurrent.py: Trains an agent using PPO2 with a recurrent policy from stable_baselines to interact with fishing-v1. Not getting optimal behavior here yet either.

dqn.py: This contains a script that runs DQN on a past fishing environment. Tried to build everything from scratch here and could not get any consistent behavior.

double_dqn.py/.ipynb: Same as above but with double DQN. Also does not work.

plot_ppo2.py: Creates a .png depicting behavior of the PPO2 trained agent from stable_baselines. 
