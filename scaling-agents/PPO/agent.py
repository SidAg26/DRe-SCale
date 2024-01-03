import os
from env import Environment as env
from stable_baselines3 import PPO


logdir = 'logs'
en = env()
models_dir = "models/${MODEL_NAME}"

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


# PPO agent from Stable Baselines 3
model = PPO(policy='MlpPolicy', env=en, n_steps=128, n_epochs=10,
            batch_size=128, tensorboard_log=logdir,
            stats_window_size=100, verbose=1) 

TIMESTEPS = 1000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, 
                reset_num_timesteps=False,
                tb_log_name="${TB_LOG_NAME}")
    model.save(f'{models_dir}/{TIMESTEPS*i}')
    print(f"Model saved at {models_dir}/{TIMESTEPS*i}")