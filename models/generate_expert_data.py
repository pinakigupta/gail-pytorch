import numpy as np
import torch
from tqdm import tqdm
from torch.nn import Module
from torch import multiprocessing
import copy
import os
from torch import multiprocessing as mp
import time
from torch.nn import Module, Sequential, Linear, Tanh, Sigmoid, Parameter, Embedding


from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from util.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch


torch.set_default_tensor_type(torch.FloatTensor)
from torch import FloatTensor



def worker(
                env_value, 
                expert, 
                data_queue, 
                episode_rewards, 
                steps_per_worker, 
                # processes_launched, 
                worker_id,
                # progress,
                lock
                ):
    env = env_value.value  # Retrieve the shared env object
    steps_collected = 0
    # expert = env.controlled_vehicles[0]

    while steps_collected < steps_per_worker:
        ob = env.reset()
        ob=ob[0]
        done = False
        ep_rwds = []
        all_obs = {}
        all_acts = {}

        # print(" Entered worker ", worker_id, " . num_steps ", steps_per_worker,  flush=True)
        while not done and steps_collected < steps_per_worker:
            # Extract features from observations using the feature extractor
            # features_extractor, policy_net, action_net = expert
            ob_tensor = copy.deepcopy(torch.Tensor(ob).to(torch.device('cpu')))
            # # print("features_extractor ", id(features_extractor), " obs ", ob_tensor.data_ptr())
            # features = expert.policy(ob_tensor)

            # # Pass observations through the MLP network
            # # policy_net = expert.policy.mlp_extractor.policy_net
            # mlp_features = expert.policy.mlp_extractor.policy_net(features)

            # # Pass features through the action network
            # # action_net = expert.policy.action_net
            # action_logits = expert.policy.action_net(mlp_features)

            # # Get the greedy action
            # greedy_action = torch.argmax(action_logits).item()
            
            act = expert.predict(ob_tensor)[0]
            # act = 0 
            next_ob, rwd, done, _, _ = env.step(act)

            obs_collected = 0

            for v in env.road.vehicles:
                if v is not env.vehicle: 
                    obs = v.observer
                    acts = env.action_type.actions_indexes[v.discrete_action]
                    if id(v) not in all_obs:
                        all_obs[id(v)] = []
                        all_acts[id(v)] = []
                    all_obs[id(v)].append(obs)
                    all_acts[id(v)].append(acts)
                    # if v.id() not in all_acts:
                    #     all_acts[v.id()] = []
                    # all_acts[v.id()].append(acts)
                    # all_acts.append(acts)
                    obs_collected += 1
            ob = next_ob
            # data_queue.put((ob, act))
            ep_rwds.append(rwd)
            steps_collected += obs_collected

        # print("all_obs ", len(all_obs))
        # Update progress value
        if lock.acquire(timeout=1):
            for (id1 , ep_obs), (id2, ep_acts) in zip(all_obs.items(), all_acts.items()):
                # print(" ep_acts length ", len(ep_acts), id1, id2, " steps_collected ", steps_collected, 
                #       " obs_collected ", obs_collected, " steps_per_worker ", steps_per_worker, " done ", done)
                data_queue.put((ep_obs, ep_acts))
            lock.release()
            # progress.value += obs_collected
            # print("worker ", worker_id, " steps_collected ", steps_collected,   flush=True)
        episode_rewards.append(np.sum(ep_rwds))
        # time.sleep(0.001)

def collect_expert_data(
                        env,
                        expert,
                        num_steps_per_iter
                        ):
    exp_rwd_iter = []
    exp_obs = []
    exp_acts = []
    torch.set_num_threads(1)
    # Create the shared Manager object
    manager = torch.multiprocessing.Manager()
    env_value = manager.Value(type(None), None)

    # Create a lock for workers
    lock = mp.Lock()
    
    processes_launched = multiprocessing.Event()

    # Initialize a queue to store expert data
    exp_data_queue = multiprocessing.Queue()

    # Initialize a list to store episode rewards
    episode_rewards = manager.list()

    # Determine the number of workers based on available CPU cores
    num_workers = max(multiprocessing.cpu_count()-5,1)

    # Calculate the number of steps per worker
    num_steps_per_worker = num_steps_per_iter // num_workers
    # num_steps_per_worker *=1.25 # allocate higher number of episodes than quota, so that free workers can do them w/o a blocking call

    # Create a list to store worker processes
    worker_processes = []

    env_value.value = env
    # Launch worker processes for expert data collection

    for i in range(num_workers):
        
        worker_process = multiprocessing.Process(
                                                    target=worker, 
                                                    args=(
                                                            copy.deepcopy(env_value), 
                                                            copy.deepcopy(expert), 
                                                            exp_data_queue, 
                                                            episode_rewards, 
                                                            num_steps_per_worker, 
                                                            # processes_launched, 
                                                            i,
                                                            # progress,
                                                            lock
                                                        )
                                                )
        worker_processes.append(worker_process)
        worker_process.start()


    pbar_outer = tqdm(total=num_steps_per_iter, desc='Progress of Expert data collection')

    # Collect expert data from the queue
    # steps_collected = 0
    while len(exp_acts) < num_steps_per_iter:
        ob, act = exp_data_queue.get()
        exp_obs.extend(ob)
        exp_acts.extend(act)
        # steps_collected+=1
        pbar_outer.update(len(exp_acts) - pbar_outer.n)
    
    print(" joining worker processes ", [worker.pid for worker in worker_processes], flush=True)


    # Join worker processes to wait for their completion
    for worker_process in worker_processes:
        worker_process.terminate()

    print(" End of worker_process join")

    # Close and join the queue
    exp_data_queue.close()
    exp_data_queue.join_thread()
    pbar_outer.close()
    # Accumulate episode rewards where episodes are done
    for rwd in episode_rewards:
        exp_rwd_iter.append(rwd)

    
    exp_rwd_mean = np.mean(exp_rwd_iter)
    print(
        "Expert Reward Mean: {}".format(exp_rwd_mean)
    )

    exp_obs = np.array(exp_obs)
    exp_acts = np.array(exp_acts)

    return exp_rwd_iter, exp_obs, exp_acts





