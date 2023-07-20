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


# Function to recursively find the last sub-layer of the last layer
def find_last_sub_layer(module):
    last_layer = module
    for layer in module.children():
        last_layer = find_last_sub_layer(layer)
    return last_layer



class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        device,
        **train_config
    ) -> None:
        super().__init__()
        self.device=device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config
        self.feature_dim = self.state_dim
        if 'features_extractor_class' in self.train_config:
            self.features_extractor = self.train_config['features_extractor_class'](
                                                                                    self.train_config['observation_space'],
                                                                                    **self.train_config['features_extractor_kwargs']
                                                                                    )
            # self.feature_dim = find_last_sub_layer(self.features_extractor).out_features # Assuming linear layer 
            self.feature_dim = 64



        self.pi =  PolicyNetwork(self.feature_dim, self.action_dim, self.discrete, device=device).to(device=device)
                           
        self.v = ValueNetwork(self.feature_dim,device=device).to(device=device)
        self.d = Discriminator(self.feature_dim, self.action_dim, self.discrete, device=device).to(device=device)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
    
        state = FloatTensor(state)
        distb = self.pi(self.obs2features(state))
        action = distb.sample().detach().cpu().numpy()

        # Retrieve the output size
        # with torch.no_grad():
        #     dummy_output = self.features_extractor(state)
        #     print(dummy_output.shape)

        return action

    def obs2features(self, obs):
        if hasattr(self, 'features_extractor'):
            return self.features_extractor(obs)
        return obs.flatten(start_dim=-2)
    
    def rollout_worker  (
                            self,
                            train_config,
                            steps_per_worker,
                            env, 
                            # progress,
                            lock,
                            rollout_data_outer_queue,
                            # processes_launched,
                            render = False
                        ):
        gae_gamma = train_config["gae_gamma"]
        horizon = train_config["horizon"]
        gae_lambda = train_config["gae_lambda"]

        # env = env_value.value  # Retrieve the shared env object
        steps = 0
        rwd_iter = []

        obs_numpy = [] 
        acts_numpy = []
        rets_numpy = []
        advs_numpy = []
        gms_numpy = []
        rwd_iter_numpy = []

        while steps < steps_per_worker:
            ep_obs = []
            ep_acts = []
            ep_rwds = []
            ep_costs = []
            ep_disc_costs = []
            ep_gms = []
            ep_lmbs = []
            progress_steps = 0 


            t = 0
            done = False

            ob = env.reset()
            ob=ob[0]
            # ob = ob.flatten()

            
            
            while not done and steps < steps_per_worker:
                
                act = self.act(ob)
                ep_obs.append(ob)
                ep_acts.append(act)
                

                if render:
                    env.render()
                ob, rwd, done, truncated, info = env.step(act)
                # ob = ob.flatten()

                ep_rwds.append(rwd)
                ep_gms.append(gae_gamma ** t)
                ep_lmbs.append(gae_lambda ** t)

                t += 1
                steps += 1
                progress_steps += 1


                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break
            
            rwd_iter.append(np.sum(ep_rwds))


            ep_obs = FloatTensor(np.array(ep_obs))
            ep_acts = FloatTensor(np.array(ep_acts))
            ep_rwds = FloatTensor(ep_rwds)
            # ep_disc_rwds = FloatTensor(ep_disc_rwds)
            ep_gms = FloatTensor(ep_gms)
            ep_lmbs = FloatTensor(ep_lmbs)

            # accessing self items

            ep_feat = self.obs2features(ep_obs)
            discriminator = self.d(ep_feat, torch.squeeze(ep_acts))
            ep_costs = (-1) * torch.log(discriminator)\
                .squeeze().detach()
            ep_disc_costs = ep_gms * ep_costs

            ep_disc_rets = FloatTensor(
                [sum(ep_disc_costs[i:]) for i in range(t)]
            )
            ep_rets = ep_disc_rets / ep_gms

            self.v.eval()
            curr_vals = self.v(ep_feat).detach()
            next_vals = torch.cat(
                (self.v(ep_feat)[1:], FloatTensor([[0.]]))
            ).detach()

            ep_deltas = ep_costs.unsqueeze(-1)\
                + gae_gamma * next_vals\
                - curr_vals

            ep_advs = FloatTensor([
                ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                .sum()
                for j in range(t)
            ])


            
            if lock.acquire(timeout=1):
                # progress[0] = progress[0] + progress_steps
                # print(" progress ", progress[0]) 
                progress_steps = 0
                rollout_data_outer_queue.put ((
                                            rwd_iter ,
                                            ep_feat.detach().numpy(),
                                            ep_acts.numpy(),
                                            ep_rets.numpy(),
                                            ep_advs.numpy(),
                                            ep_gms.numpy()  
                                        ))
                obs_numpy = []
                acts_numpy = []
                rets_numpy = []
                advs_numpy = []
                gms_numpy = []
                rwd_iter_numpy = []
                lock.release()
            # else:
            #     obs_numpy.extend(ep_obs.numpy()) 
            #     acts_numpy.extend(ep_acts.numpy())
            #     rets_numpy.extend(ep_rets.numpy())
            #     advs_numpy.extend(ep_advs.numpy())
            #     gms_numpy.extend(ep_gms.numpy())
            #     rwd_iter_numpy.append(rwd_iter)
            #     print("acts_numpy length ", len(acts_numpy),   flush=True)
            

            

    def rollout(self,env, num_steps_per_iter, iteration, render=False):
        rwd_iter = []
        obs = []
        acts = []
        rets = []
        advs = []
        gms = []

        torch.set_num_threads(1)
        lock = mp.Lock()

         # Determine the number of workers based on available CPU cores
        num_workers = 50 #multiprocessing.cpu_count()
        manager = torch.multiprocessing.Manager()
        # progress = manager.Value('i', 0)
        progress = [0]

        processes_launched = multiprocessing.Event()

        # env_value = manager.Value(type(None), None)
        # env_value.value = env
        # Calculate the number of steps per worker
        num_steps_per_worker = num_steps_per_iter // num_workers

        # Initialize a queue to store expert data
        rollout_data_outer_queue = multiprocessing.Queue()

        # Create a list to store worker processes
        worker_processes = []
        train_config = copy.deepcopy(self.train_config)

        for i in range(num_workers):
            
            worker_process = multiprocessing.Process(
                                                        target=self.rollout_worker, 
                                                        args=(  
                                                                train_config,
                                                                num_steps_per_worker, 
                                                                copy.deepcopy(env), 
                                                                # progress,
                                                                lock,
                                                                rollout_data_outer_queue,
                                                                # processes_launched
                                                            )
                                                    )
            worker_processes.append(worker_process)
            worker_process.start()


        pbar = tqdm(total=num_steps_per_iter, desc='Progress rollouts for iteration '+str(iteration))
        while len(acts) < num_steps_per_iter :
            
            rwd_iter_q, obs_q, acts_q , rets_q, advs_q, gms_q = rollout_data_outer_queue.get()
            obs.extend(obs_q)
            acts.extend(acts_q)
            rwd_iter.extend(rwd_iter_q)
            rets.extend(rets_q)
            advs.extend(advs_q)
            gms.extend(gms_q)
            pbar.update(len(acts) - pbar.n)

        print(" joining worker processes ", [worker.pid for worker in worker_processes], flush=True)
        # Join worker processes to wait for their completion
        for worker_process in worker_processes:
            worker_process.join()


        pbar.close()


        obs =  FloatTensor(np.array(obs))
        # obs = obs.flatten(start_dim=0, end_dim=1)
        acts = FloatTensor(np.array(acts))
        rets = FloatTensor(np.array(rets)) 
        advs = FloatTensor(np.array(advs)) 
        gms =  FloatTensor(np.array(gms)) 

        return rwd_iter, obs, acts, rets, advs, gms

    def train(self, env, exp_obs, exp_acts, render=False):

        optimal_agent = "optimal_gail_agent.pth"
        optimal_reward = -float('inf')
        if os.path.exists(optimal_agent):
            os.remove(optimal_agent)

        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        num_expert_steps = self.train_config["num_expert_steps"]
        
        lambda_ = self.train_config["lambda"]
        
        
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_d = torch.optim.Adam(self.d.parameters())


        
        # exp_obs = self.features_extractor(exp_obs).detach()
        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter, obs, acts, rets, advs, gms = self.rollout(
                                                                 env, 
                                                                 num_steps_per_iter, 
                                                                 i,
                                                                 render=render
                                                                 )

            rwd_iter_mean = np.mean(rwd_iter)
            rwd_iter_means.append(rwd_iter_mean)
            print(
                "Iterations: {},   Reward Mean: {}, rwd_iter_size: {}, acts_size: {}"
                .format(i + 1, rwd_iter_mean, len(rwd_iter), len(acts)))
            

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            # obs = self.features_extractor(obs.reshape(-1,10, 7))
            
            if hasattr(self, 'features_extractor'):
                self.features_extractor.train()
            self.d.train()

            exp_feat = self.obs2features(exp_obs)
            exp_scores = self.d.get_logits(exp_feat, exp_acts)
            nov_scores = self.d.get_logits(obs, torch.squeeze(acts))

            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )
            loss.backward()
            opt_d.step()

            self.v.train()
            
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

            if not os.path.exists(optimal_agent):
                torch.save(self.state_dict(), optimal_agent)
            elif rwd_iter_mean > 1.1*optimal_reward and rwd_iter_mean > 0 :
                optimal_reward=rwd_iter_mean
                print(" saving optimal gail agent with reward ", optimal_reward)
                torch.save(self.state_dict(), optimal_agent)

        torch.save(self.state_dict(), 'gail_agent.pth')
        return rwd_iter_means, optimal_agent
        # return exp_rwd_mean, rwd_iter_means


