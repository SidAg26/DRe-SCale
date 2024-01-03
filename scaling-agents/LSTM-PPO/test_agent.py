import numpy as np
import os
from test_env import Environment as env
from sb3_contrib import RecurrentPPO
import tensorflow as tf


logdir = 'logs/evaluation/'
en = env()
models_dir = "models/evaluation/${MODEL_NAME}"

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


# Load the model
model = RecurrentPPO.load(path='models/${MODEL_NAME}/${MODEL}.zip',
                          env=en,
                          custom_objects={'clip_range': 0.2,'lr_schedule': 1})


episodes = 500
rewards_history = []
avg_score = 0
_states = None # Necessary step for LSTM integration
num_envs = 1
episode_starts = np.ones((num_envs,), dtype=bool)
for ep in range(episodes):
    obs, info = en.reset()
    done = False
    ep_rew = 0    
    while not done:
        action, _states = model.predict(obs, state=_states)
        obs, rewards, done, _, info = en.step(action.item(0))
        ep_rew += rewards
        episode_starts = done
    rewards_history.append(ep_rew) 
    avg_score = np.mean(rewards_history[-100:])   
    tf.summary.scalar('reward summary', data=avg_score, step=ep)
    tf.summary.scalar('episodic_reward', ep_rew, ep)
