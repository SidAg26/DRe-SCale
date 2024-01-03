from datetime import datetime
import time as time
import math
import json as json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf

from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

# Utility to connect to K8s API, Prometheus API and OpenFaaS API
config.load_config()
deployment_name = 'matmul' # deployment name for deployed function
namespace = 'openfaas-fn' # default namespace for openfaas functions
scale_api = client.AppsV1Api()
resource_usage_api = client.CustomObjectsApi()
prom = PrometheusConnect(url='${PROMETHEUS_URL}', disable_ssl=True)


class Environment(gym.Env):

    # every environment should support None render mode
    metadata = {'render_modes': ['human', None]}

    def __init__(self, rew_range=(-100, 10000), min_pods=1, max_pods=24) -> None:
        super(Environment, self).__init__()

        
        # FIXED PARAMETERS / Configurable
        self.reward_range = rew_range
        self.MAX_PODS = max_pods
        self.MIN_PODS = min_pods

        self.sampling_window = 30 
        self.timestep = 0
        self.episode = 0
        self.loop = 0
        self._last_obs = None
        self._stats_window = 100

        self.reward_history = []
        self.score = 0
        # [avg_execution, throughput, requests, replicas, avg_cpu/req, avg_mem/req]
        self.observation_space = spaces.Box(
                                    low=np.array([0, 0, 0, self.MIN_PODS, 0, 0]), 
                                    high=np.array([60, 100, 100, self.MAX_PODS, 2, 2]), 
                                    shape=(6,), 
                                    dtype=np.float64)
        
        # action is either increase or decrease pods
        self.action_space = spaces.Discrete(5)
        self._action_to_scale = {0: -2,
                                 1: -1,
                                 2: 0,
                                 3: 1,
                                 4: 2}
        self._initial_setup()

        
 

    def _initial_setup(self):
        # function resource requests
        self.func_cpu = 150 # millicores
        self.func_mem = round((256/1024), 2) # in GBi

        # custom metrics from env
        logdir = "logs/evaluation/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(logdir + "/${MODEL_NAME}")
        self.file_writer.set_as_default() 

        self._reward_file = 'evaluation_reward_history_${MODEL_NAME}.json'

    def _get_info(self):
        return {}
    
    # Utility function to take action in the environment
    def _take_action(self, action):
        try:
            # number of ready function replicas 
            current_pods = scale_api.read_namespaced_deployment(name=deployment_name,
                                            namespace=namespace).status.ready_replicas
            if current_pods == None:
                current_pods = 0
                print('current pods are NoneType')
        except Exception as e:
            current_pods = 0
            print('Error in reading ready pods')

        scale_value = current_pods + action
        action_feedback = False

        if action < 0 :
            if (scale_value >= self.MIN_PODS):
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    _ = scale_api.patch_namespaced_deployment_scale(name=deployment_name, 
                                                    namespace=namespace,
                                                    body=body).spec.replicas
                except Exception as e:
                    action_feedback = False
                    print(e)
            else:
                action_feedback = False

        elif action == 0:
            if scale_value == 0:
                action_feedback = True
            else:
                action_feedback = False

        else:
            if (scale_value <= self.MAX_PODS and scale_value >= self.MIN_PODS):
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    _ = scale_api.patch_namespaced_deployment_scale(name=deployment_name, 
                                                        namespace=namespace,
                                                        body=body).spec.replicas
                except Exception as e:
                    action_feedback = False
                    print(e)
            else:
                action_feedback = False
             
        info = {'action': action, 
               'action_feedback': action_feedback, 
               'pods': current_pods,
               'scale_value': scale_value}
        
        return info

    def _get_obs(self):
        obs = None

        # get the avg execution time
        query1 = "(rate(gateway_functions_seconds_sum{function_name='matmul.openfaas-fn',\
              code='200'}[30s]) / \
                rate(gateway_functions_seconds_count{function_name='matmul.openfaas-fn',\
                      code='200'}[30s]))"
       
        try:
            data = prom.custom_query(query=query1)
            avg_execution = round(float((data[0]['value'][1])), 3)
            avg_execution = {True:0, False: avg_execution}[math.isnan(avg_execution)]
        except Exception as e:
            avg_execution = 0.0

        try:
            # can be obtained from gateway_service_count
            query3 = "kube_deployment_status_replicas_ready{deployment='matmul'}"
            data = prom.custom_query(query=query3)
            replicas = int(float(data[0]['value'][1]))
        except Exception as e:
            replicas = 0
            print(e)


        try:
            # total requests during the period
            query4 = "increase(gateway_function_invocation_total{function_name='matmul.openfaas-fn'}[30s])"
            data = prom.custom_query(query=query4)
            total = 0
            for d in data:
                total += int(float(d['value'][1]))
            requests = total
        except Exception as e:
            requests = 0
        print(f'requests are {requests}')

        try:
            # throughput during the period (percentage)
            query2 = "increase(gateway_function_invocation_total{code='200', function_name='" + deployment_name + "." + namespace + "'}[30s])"
            data = prom.custom_query(query=query2)
            throughput = int(float(data[0]['value'][1]))
            throughput = int(round((throughput/requests)*100, 2))
        except ZeroDivisionError:
            if requests == 0:
                throughput = 100
            else:
                throughput = 0
        except Exception:
            if requests == 0:
                throughput = 100
            else:
                throughput = 0
      

        try:
        # get the avg usage metrics
            resource_list = resource_usage_api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", "openfaas-fn", "pods")
            my_pods  = [pod['containers'][0]['usage'] for pod in resource_list['items'] if pod['metadata']['labels']['faas_function'] == deployment_name]
            cpu = 0
            mem = 0
            for pods in my_pods:
                c = pods['cpu']
                m = pods['memory']
                try:
                    # converting everything in to millicores (m) 1 vCPU = 1000m
                    if c.endswith('n'):
                        cpu += (round(int(c.split('n')[0])/1e6, 4))
                    elif c.endswith('u'):
                        cpu += (round(int(c.split('u')[0])/1e3, 4))
                    elif c.endswith('m'):
                        cpu += (round(int(c.split('m')[0]), 4))
                    else:
                        cpu += 0
                except Exception as e:
                    cpu += 0
                try:    
                    # converting everything into Gi
                    if m.endswith('Ki'):
                        mem += (round(int(m.split('Ki')[0])/(1024*1024), 4))
                    elif m.endswith('Mi'):
                        mem += (round(int(m.split('Mi')[0])/1024, 4))
                    elif m.endswith('Gi'):
                        mem += (round(int(m.split('Gi')[0]), 4))
                    else:
                        mem += 0
                except Exception as e:
                    mem += 0
            avg_cpu = round((cpu/len(my_pods))/self.func_cpu, 4)
            avg_mem = round((mem/len(my_pods))/self.func_mem, 4)

        except Exception as e:
            print('pods not available for metrics')
            # if len(my_pods) == 0: # case where pods are unavailable
            my_pods = 0
            avg_cpu = 0
            avg_mem = 0
            
        # get the next observation from the environment after action
        obs = np.array([avg_execution, throughput, requests, replicas, avg_cpu, avg_mem])

        return obs
        
    def _write_to_board(self, obs, action, rew, info, step, episode):
        # write to tensorboard
        with self.file_writer.as_default():
            tf.summary.scalar('avg_execution_time', obs[0], step)
            tf.summary.scalar('throughput', obs[1], step)
            tf.summary.scalar('requests', obs[2], step)
            tf.summary.scalar('replicas', obs[3], step)
            tf.summary.scalar('cpu', obs[4], step)
            tf.summary.scalar('mem', obs[5], step)
            tf.summary.scalar('episode', episode, step)
            tf.summary.scalar('action', (action), step)
            if info['action_feedback']:
                tf.summary.scalar('action_feedback', 1 , step)
            else:
                tf.summary.scalar('action_feedback', 0 , step)
            tf.summary.scalar('n-step_reward', rew, step)


    # calculate and return reward based on the observation
    def _calculate_reward(self, obs, metadata={}):
        reward = 0
        meta_scale_value = metadata['scale_value']
        throughput = obs[1] # %
        _ = obs[2]
        replicas = obs[3]
        avg_cpu = obs[4] # % 0 - 1
        avg_mem = obs[5] # % 0 - 1


        alpha = 0.75
        beta = 0.125
        gamma = 0.125
        phi = 0.25
        r_th = alpha * (throughput ** 2)
        r_cpu = beta * (avg_cpu*100)
        r_mem = gamma * (avg_mem*100)
        r_rep = -phi * ((replicas - self.MIN_PODS) ** 2)

        reward = r_th + r_cpu + r_mem + r_rep
        reward = round(reward, 2)

        # action unsuccessful
        if (meta_scale_value != replicas):
            reward += self.reward_range[0]
        
        return reward
    


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # reset other paramters based on the environment
        self.score = 0
        self.loop = 0
        observation = self._get_obs()
        info = self._get_info()
        self._last_obs = observation
      
        return observation, info


    def step(self, action):
        done = False
        # Map the action (element of {0,1,2,3,4}) to scaling
        action = self._action_to_scale[action]

        # execute the action in environment
        info = self._take_action(action=action)

        # immediate negative reward - invalid action
        if info['action_feedback'] == False:
            self._write_to_board(self._last_obs, action, -100, info, self.timestep, self.episode)
            self.timestep += 1
            self.loop += 1
            self.score += -100
            if self.loop == 10:
                done = True
            return self._last_obs, -100, done, False, info 
        else:
            # wait for the sampling window to get the next observation
            time.sleep(self.sampling_window)

            # get the next observation
            next_obs = self._get_obs()
            # calculate reward
            reward = self._calculate_reward(obs=next_obs, metadata=info)
            self.score += round(reward, 2)
            self._write_to_board(next_obs, action, reward, info, self.timestep, self.episode)

            # counter for custom metrics
            self.timestep += 1
            if (self.timestep % 10 == 0):
                done = True
                self.episode += 1
                self.loop = 0
                self.reward_history.append(self.score)
                with self.file_writer.as_default():
                    tf.summary.scalar('episodic_reward', self.score, self.episode)
                    tf.summary.scalar('mean_reward', np.mean(self.reward_history[-self._stats_window:]), self.episode)
                self.score = 0
                history = {'reward_history': self.reward_history,
                              'last_episode': self.episode}
                # write the reward history to a file
                with open(self._reward_file, "w") as outfile:
                    json.dump(history, outfile)
                
            self._last_obs = next_obs

            return next_obs, reward, done, False, info
    
    
    def render(self, mode='human', close=False):
        # render or print information on screen or add to the tensorboard, etc.
        pass

    def close(self):
        # close any open resources
        pass