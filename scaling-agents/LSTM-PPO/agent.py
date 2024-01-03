import os
from env import Environment as env
from sb3_contrib import RecurrentPPO


logdir = 'logs'
en = env()
models_dir = "models/${MODEL_NAME}"

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# PPO agent with LSTM
model = RecurrentPPO("MlpLstmPolicy", 
                     env=en, verbose=1, 
                     n_steps=128, n_epochs=10,
                     stats_window_size=100, 
                     batch_size=128,
                     policy_kwargs={'n_lstm_layers': 1, 
                                    'lstm_hidden_size': 256,
                                    'net_arch': dict(pi=[64, 64], vf=[64, 64])},
                     tensorboard_log=logdir
                     )

TIMESTEPS = 1000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, 
                reset_num_timesteps=False,
                tb_log_name="${TB_LOG_NAME}")
    model.save(f'{models_dir}/{TIMESTEPS*i}')
    print(f"Model saved at {models_dir}/{TIMESTEPS*i}")
