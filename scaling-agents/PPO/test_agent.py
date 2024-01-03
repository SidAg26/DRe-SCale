
import os
import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO
import stable_baselines3.common.vec_env
from test_env import Environment as env


logdir = 'logs/evaluation'
en = env()
models_dir = "models/${MODEL_NAME}"

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


en = env()
en = stable_baselines3.common.vec_env.DummyVecEnv([lambda: en])
en.reset()
model = PPO.load(model_path='{models_dir}/{MODEL}.zip', env=en) # Load the PPO model

episodes = 50
rewards_history = []
for ep in range(episodes):
    obs = en.reset()
    done = False
    ep_rew = 0
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, info = en.step(action)
        ep_rew += rewards[0]
    rewards_history.append(ep_rew) 
    avg_score = np.mean(rewards_history[-100:])
    tf.summary.scalar('reward summary', data=avg_score, step=ep)
    tf.summary.scalar('episodic_reward', ep_rew, ep)