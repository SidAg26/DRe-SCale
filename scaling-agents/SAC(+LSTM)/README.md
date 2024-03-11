## SAC (Soft-Actor-Critic) with LSTM Agent for AutoScaling Functions
-   `sac_agent.py` and `sac_lstm_agent.py` contains the agent training code 
-   `env.py` contains the `gymnasium` supported integrated Kubernetes/OpenFaaS environment for interaction and feedback loop
-   `--train` & `--test` flags for respective train and test actions of the agent
-   `requirements.txt` contains the project requirements


<br>

__Note:__ <br>
-   To successfully run the agent please update the relevent placeholders marked with `$PLACEHOLDER`.
-   Code Reference: <br>
```
@misc{rlalgorithms,
  author = {Zihan Ding},
  title = {Popular-RL-Algorithms},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/quantumiracle/Popular-RL-Algorithms}},
}
```
