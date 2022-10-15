from operator import attrgetter
import torch

import numpy
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import gym
#import mujoco_py
import numpy as np
from gym.spaces import Box, Discrete
import setup
import math
from poison_rl.memory import Memory
from poison_rl.agents.vpg import VPG
from poison_rl.agents.ppo import PPO
from poison_rl.attackers.wb_attacker import WbAttacker
from poison_rl.attackers.fgsm_attacker import FGSMAttacker
from poison_rl.attackers.targ_attacker import TargAttacker
from poison_rl.attackers.bb_attacker import BbAttacker
from poison_rl.attackers.rand_attacker import RandAttacker
from torch.distributions import Categorical, MultivariateNormal

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--run', type=int, default=-1)
# env settings
parser.add_argument('--env', type=str, default
="CartPole-v0")
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--steps', type=int, default=300)

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type = float, default = 0.99)

# Federated settings
parser.add_argument('--agents',type = int, default=2) # number of agents
parser.add_argument('--rounds', type = int, default = 10) # number of communication rounds


# attack settings
parser.add_argument('--norm', type=str, default="l2")
parser.add_argument('--stepsize', type=float, default=0.05)
parser.add_argument('--maxiter', type=int, default=10)
parser.add_argument('--radius', type=float, default=0.3)
parser.add_argument('--radius-s', type=float, default=0.1)
parser.add_argument('--radius-a', type=float, default=0.3)
parser.add_argument('--radius-r', type=float, default=0.05)
parser.add_argument('--frac', type=float, default=1)
# parser.add_argument('--type', type=str, default="wb", help="rand, wb, semirand")
parser.add_argument('--type', type=str, default="targ", help="rand, wb, semirand")


parser.add_argument('--aim', type=str, default="reward", help="reward, obs, action")

parser.add_argument('--attack', dest='attack', action='store_true')
parser.add_argument('--no-attack', dest='attack', action='store_false')
parser.set_defaults(attack=True)

parser.add_argument('--compute', dest='compute', action='store_true')
parser.add_argument('--no-compute', dest='compute', action='store_false')
parser.set_defaults(compute=False)

# file settings
parser.add_argument('--logdir', type=str, default="logs/")
parser.add_argument('--resdir', type=str, default="results/")
parser.add_argument('--moddir', type=str, default="models/")
parser.add_argument('--loadfile', type=str, default="")





args = parser.parse_args()

# def get_log(file_name):
#     logger = logging.getLogger('train') 
#     logger.setLevel(logging.INFO) 

#     fh = logging.FileHandler(file_name, mode='a') 
#     fh.setLevel(logging.INFO) 
    
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)  
#     return logger


def Agent(attack_flag = False, random_seed = 0, round_index = 1, agent_index = 1):
    
    # Local training;
    if attack_flag:
        print("Poisoning...")
    else:
        print("Clean Local Training..."+ "Agent "+str(agent_index))
    
    ############## Hyperparameters ##############
    env_name = args.env #"LunarLander-v2"
    
    max_episodes = args.episodes        # max training episodes
    max_steps = args.steps         # max timesteps in one episode
    
    attack = attack_flag
  
    compute = args.compute
    attack_type = args.type
    learner = args.learner
    aim = args.aim
    
    stepsize = args.stepsize
    maxiter = args.maxiter
    radius = args.radius
    frac = args.frac
    lr = args.lr
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ############ For All #########################
    gamma = args.gamma                # discount factor
    random_seed = random_seed 
    render = False
    update_every = 300
    save_every = 100
    
    ########## creating environment
    env = gym.make(env_name)
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    
    ########## file related 
    filename = "RE_"+env_name + "_" + learner + "_A" + str(args.agents) + "_C"+ str(args.rounds)+"_n" + str(max_episodes) 
    if attack:
        filename += "_" + attack_type + "_" + aim
        filename += "_s" + str(stepsize) + "_m" + str(maxiter) + "_r" + str(radius) + "_f" + str(frac) + "_c" +str(round_index)
    else:
        filename += "_" + "clean" + "_a"+ str(agent_index) + "_c" + str(round_index)
    
    if args.run >=0:
        filename += "_run" + str(args.run)
        
        
    # logger = get_log(args.logdir + filename + "_" +current_time)
    # logger.info(args)
    
    rew_file = open(args.resdir + filename + ".txt", "w")
    if attack_type == "targ" or attack_type == "fgsm":        
        targ_file = open(args.resdir + filename + "_targ.txt", "w")
        targ_metrix = []
    if compute:
        radius_file = open(args.resdir + filename + "_radius" + "_s" + str(stepsize) + "_m" + str(maxiter) + "_r" + str(radius) + ".txt", "w")
    
    
    ########## create learner
    if learner == "vpg":
        policy_net = VPG(env.observation_space, env.action_space, gamma=gamma, device=device, learning_rate=lr)
    elif learner == "ppo":
        policy_net = PPO(env.observation_space, env.action_space, gamma=gamma, device=device, learning_rate=lr)
    
    
    ########## create attacker
    if attack_type == "wb":
        attack_net = WbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, device=device)
    elif attack_type == "bb":
        attack_net = BbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, device=device)
    elif attack_type == "rand":
        attack_net = RandAttacker(env.action_space, radius=radius, frac=frac, maxat=int(frac*max_episodes), device=device)
    elif attack_type == "semirand":
        attack_net = WbAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, rand_select=True)
    elif attack_type == "targ":
        if isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
            target_policy = action_dim - 1
        elif isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
            target_policy = torch.zeros(action_dim)
#            target_policy[-1] = 1
        if attack_flag:
            print("target policy is", target_policy)
        
        if attack:
            attack_net = TargAttacker(env.observation_space, env.action_space, policy_net, maxat=int(frac*max_episodes), maxeps=max_episodes,
                                targ_policy=target_policy, gamma=gamma, learning_rate=lr, maxiter=maxiter, radius=radius, stepsize=stepsize, device=device)
    elif attack_type == "fgsm":
        if isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
            target_policy = action_dim - 1
        elif isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
            target_policy = torch.zeros(action_dim)
        def targ_policy(obs):
            return target_policy
        attack_net = FGSMAttacker(policy_net, env.action_space, targ_policy, radius=radius, frac=frac, maxat=int(frac*max_episodes), device=device)
    
    if aim == "obs" or aim == "hybrid":
        attack_net.set_obs_range(env.observation_space.low, env.observation_space.high)
    
    start_episode = 0
    # train starting from the coordinator's broadcast model
    if args.loadfile != "":
        checkpoint = torch.load(args.moddir + args.loadfile)
        print("load from ", args.moddir + args.loadfile)
        policy_net.set_model_state_dict(checkpoint['model_state_dict'])
        # policy_net.set_state_dict(checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'])
        # start_episode = checkpoint['episode']
    
    memory = Memory()
    
    all_rewards = []
    timestep = 0
    update_num = 0
    
    ######### training
    for episode in range(start_episode, max_episodes):
        state = env.reset()
        if len(state)!=1:
            state = state[0]
        rewards = []
        total_targ_actions = 0
        for steps in range(max_steps):
            timestep += 1
            
            if render:
                env.render()
                
            state_tensor, action_tensor, log_prob_tensor = policy_net.act(state)
            
            if isinstance(env.action_space, Discrete):
                action = action_tensor.item()
                if attack_type == "targ" or attack_type == "fgsm":
                    if action == target_policy:
                        total_targ_actions += 1
            else:
                action = action_tensor.cpu().data.numpy().flatten()
                if attack_type == "targ" or attack_type == "fgsm":
                    total_targ_actions += np.linalg.norm(action - target_policy.numpy()) ** 2
#            print(action, target_policy, total_targ_actions)
                
            new_state, reward, done,_, _ = env.step(action)
            
            if attack_type == "fgsm":
#                before_attack = new_state.copy()
#                print("before attack", new_state)
                new_state = attack_net.attack(new_state)
#                print("after attack", new_state)
#                print("attack norm", np.linalg.norm(before_attack - new_state))
            
            rewards.append(reward)
            
            memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
            
            if done or steps == max_steps-1: #timestep % update_every == 0:
                if attack and attack_type != "fgsm":
                    if aim == "reward":
                        attack_r = attack_net.attack_r_general(memory)
                        # logger.info(memory.rewards)
                        memory.rewards = attack_r.copy()
                        # logger.info(memory.rewards)
                    elif aim == "obs":
                        attack_s = attack_net.attack_s_general(memory)
                        # logger.info(torch.stack(memory.states).to(device).detach().cpu().numpy().tolist())
                        memory.states = attack_s.copy()
                        # logger.info(torch.stack(memory.states).to(device).detach().cpu().numpy().tolist())
                    elif aim == "action":
                        attack_a = attack_net.attack_a_general(memory)
                        # logger.info(torch.stack(memory.actions).to(device).detach().cpu().numpy().tolist())
                        memory.actions = attack_a.copy()
                        # logger.info(torch.stack(memory.actions).to(device).detach().cpu().numpy().tolist())
                    elif aim == "hybrid":
                        res_aim, attack = attack_net.attack_hybrid(memory, args.radius_s, args.radius_a, args.radius_r)
                        print("attack ", res_aim)
                        if res_aim == "obs":
                            # logger.info(memory.states)
                            memory.states = attack.copy()
                            # logger.info(memory.states)
                        elif res_aim == "action":
                            # logger.info(memory.actions)
                            memory.actions = attack.copy()
                            # logger.info(memory.actions)
                        elif res_aim == "reward":
                            # logger.info(memory.rewards)
                            memory.rewards = attack.copy()
                            # logger.info(memory.rewards)
                    if attack_type == "bb": # and attack_net.buffer.size() > 128:
                        attack_net.learning(memory)
#                    print("attacker")
#                    attack_net.print_paras(attack_net.im_policy)
                if compute:
                    stable_radius = attack_net.compute_radius(memory)
                    print("stable radius:", stable_radius)
                    radius_file.write("episode: {}, radius: {}\n".format(episode, np.round(stable_radius, decimals = 3)))
                if attack_type == "targ" or attack_type == "fgsm":
                    if isinstance(env.action_space, Discrete):
                        targ_file.write(str(float(total_targ_actions) / (steps+1)) + "\n")
                        targ_metrix.append(float(total_targ_actions) / (steps+1))
                        # print("percent of target", float(total_targ_actions) / (steps+1))
                    else:
                        targ_file.write(str(math.sqrt(total_targ_actions / (steps+1))) + "\n")
                        targ_metrix.append(float(total_targ_actions) / (steps+1))
                        # print("average distance to target", math.sqrt(total_targ_actions / (steps+1)))
                policy_net.update_policy(memory)
#                print("learner")
#                attack_net.print_paras(policy_net.policy)
                memory.clear_memory()
                timestep = 0
                update_num += 1
                
            state = new_state
            
            if done or steps == max_steps-1:
                all_rewards.append(np.sum(rewards))
                # logger.info("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                rew_file.write("episode: {}, total reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                break
        
        if (episode+1) % save_every == 0 and attack_type != "rand" and attack_type != "fgsm":
            path = args.moddir + filename
            
            if attack:
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy_net.policy.state_dict(),
                    'optimizer_state_dict': policy_net.optimizer.state_dict(),
                    'attack_critic': attack_net.critic.state_dict(),
                    'attack_critic_optim': attack_net.critic_optim.state_dict()
                    }, path)
            else: 
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy_net.policy.state_dict(),
                    'optimizer_state_dict': policy_net.optimizer.state_dict(),
                    # 'attack_critic': attack_net.critic.state_dict(),
                    # 'attack_critic_optim': attack_net.critic_optim.state_dict()
                    }, path)

            print("Episode", episode+1 ,"/",max_episodes)
    if attack:
        # logger.info("total attacks: {}\n".format(attack_net.attack_num))
        print("total attacks: {}\n".format(attack_net.attack_num))
        print("update number:", update_num)
        
    rew_file.close()

    if attack_type == "targ" or attack_type == "fgsm":
        targ_file.close()

    if compute:
        radius_file.close()
    env.close()
    
    return targ_metrix        








def aver_model_dict(dict_list):
    # generate an average dict from a dict list
    res = {}
    for k in dict_list[0].keys():
        res[k] = torch.mean(torch.stack([i_dict[k] for i_dict in dict_list]), dim = 0)
    
    return res



def Plot_Poison_Single_RL():
    # ###################################################################################
    # Oct 6 replication for poisoning on single agent.
    Targ_metrix_targ = Agent(attack_flag=True, random_seed=0, round_index = 1, agent_index=0)
    Targ_metrix_clean = Agent(attack_flag=False, random_seed=0, round_index = 1, agent_index = 1)
       
    plt.xlabel("Episode")
    plt.ylabel("Propotion of target actions")
    plt.title("Target Poisoning")
    plt.plot(np.arange(args.episodes)+1,Targ_metrix_targ, c="red", label = "Poison")
    plt.plot(np.arange(args.episodes)+1,Targ_metrix_clean, c="green", label = "Poison")
    plt.show()



def Poison_Fed_RL():
    for i_round in range(args.rounds):
        print("Federated Round", i_round+1)
        Malicious_targ_act_frac = Agent(attack_flag=True, random_seed=0, round_index = i_round+1, agent_index = 0) # Malicious agent local train
        
        for i_agent in range(args.agents-1):
            # Clean agents local train
            Clean_targ_act_frac = Agent(attack_flag=False, random_seed=0, round_index = i_round+1, agent_index = i_agent+1)
        
        # Coordinator model dir
        CO_filename = "RE_CO_" + args.env + "_" + args.learner + "_A" + str(args.agents) + "_C"+ str(args.rounds)+ "_n" + str(args.episodes) + "_c"+str(i_round+1) + "_a" + str(args.agents)
        CO_filename += "_" + args.type + "_" + args.aim + "_s" + str(args.stepsize) + "_m" + str(args.maxiter) + "_r" + str(args.radius) + "_f" + str(args.frac)
        # save for each communication round 
        args.loadfile = CO_filename # update the broadcast dir for next round of local training

        # Load all local models for Coordinator 
        ALL_model_state_dict = []
        # ALL_optimizer_state_dict = []

        filename = "RE_"+args.env + "_" + args.learner+ "_A" + str(args.agents) + "_C"+ str(args.rounds) + "_n" + str(args.episodes)  
        
        for i_agent in range(args.agents): # All agents
            if i_agent == 0: # load poisoned model
                i_filename = filename + "_" + args.type + "_" + args.aim + "_s" + str(args.stepsize) + "_m" + str(args.maxiter) + "_r" + str(args.radius) + "_f" + str(args.frac)+"_c" + str(i_round+1)
            else: # load clean model
                i_filename = filename + "_" + "clean" + "_a"+ str(i_agent)+ "_c" + str(i_round+1)
            i_checkpoint = torch.load(args.moddir + i_filename)
        
            print("load from ", args.moddir + i_filename)
            
            ALL_model_state_dict.append(i_checkpoint['model_state_dict'] )
            # ALL_optimizer_state_dict.append(i_checkpoint['optimizer_state_dict'])
        
        # update coordinator's model
        CO_model_state_dict = aver_model_dict(ALL_model_state_dict)
        # CO_optimizer_state_dict = aver_optimizer_dict(ALL_optimizer_state_dict)

        torch.save({
                    'model_state_dict': CO_model_state_dict
                    # 'optimizer_state_dict': CO_optimizer_state_dict
                        }, args.moddir + CO_filename)




def Evaluate_Coordinator(n_round = 5, random_seed = 0, attack_flag = 1):
    for i_round in range(n_round):
        ###### Load model
        i_CO_filename = "RE_CO_" + args.env + "_" + args.learner+ "_A" + str(args.agents) + "_C"+ str(args.rounds) + "_n" + str(args.episodes) + "_c"+str(i_round+1) + "_a" + str(args.agents)
        i_CO_filename += "_" + args.type + "_" + args.aim + "_s" + str(args.stepsize) + "_m" + str(args.maxiter) + "_r" + str(args.radius) + "_f" + str(args.frac)    
        i_checkpoint = torch.load(args.moddir + i_CO_filename)
        print("load from ", args.moddir +  i_CO_filename)
        
        ##### activate environment
        env = gym.make(args.env)
        if random_seed:
            torch.manual_seed(random_seed)
            env.seed(random_seed)

        if args.type == "targ":
            if isinstance(env.action_space, Discrete):
                action_dim = env.action_space.n
                target_policy = action_dim - 1
            elif isinstance(env.action_space, Box):
                action_dim = env.action_space.shape[0]
                target_policy = torch.zeros(action_dim)
    #            target_policy[-1] = 1
            if attack_flag:
                print("target policy is", target_policy)
        
        ########## create learner
        if args.learner == "vpg":
            policy_net = VPG(env.observation_space, env.action_space, gamma=args.gamma, device=args.device, learning_rate=args.lr)
        elif args.learner == "ppo":
            policy_net = PPO(env.observation_space, env.action_space, gamma=args.gamma, device=args.device, learning_rate=args.lr)
        
        policy_net.set_model_state_dict(i_checkpoint['model_state_dict'])
        
        ##### Link result file for target action fraction
        i_targ_file = open(args.resdir + i_CO_filename +"_targ" +".txt", "w")
        
        ##### Testing      
        start_episode = 0
        save_every = 100
        # memory = Memory()
        all_rewards = []
        timestep = 0
        update_num = 0
        ######### training
        for episode in range(start_episode, args.episodes):
            state = env.reset()
            if len(state)!=1:
                state = state[0]
            # rewards = []
            total_targ_actions = 0
            for steps in range(args.steps):
                timestep += 1
                state_tensor, action_tensor, log_prob_tensor = policy_net.act(state)
                
                if isinstance(env.action_space, Discrete):
                    action = action_tensor.item()
                    if args.type == "targ" or args.type == "fgsm":
                        if action == target_policy:
                            total_targ_actions += 1
                else:
                    action = action_tensor.cpu().data.numpy().flatten()
                    if args.type == "targ" or args.type == "fgsm":
                        total_targ_actions += np.linalg.norm(action - target_policy.numpy()) ** 2
    #            print(action, target_policy, total_targ_actions)
                    
                new_state, reward, done,_, _ = env.step(action)

                # rewards.append(reward)
                
                # memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                
                if done or steps == args.steps-1: #timestep % update_every == 0:
                    if args.type == "targ" or args.type == "fgsm":
                        if isinstance(env.action_space, Discrete):
                            i_targ_file.write(str(float(total_targ_actions) / (steps+1)) + "\n")
                        else:
                            i_targ_file.write(str(math.sqrt(total_targ_actions / (steps+1))) + "\n")
                            # print("average distance to target", math.sqrt(total_targ_actions / (steps+1)))
                    # policy_net.update_policy(memory)
                    # memory.clear_memory()
                    timestep = 0
                    update_num += 1
                    break
                    
                state = new_state
            
            if (episode+1) % save_every == 0 and args.type != "rand" and args.type != "fgsm":
                print("Episode", episode+1 ,"/",args.episodes)
            
        i_targ_file.close()
        env.close()
      



        
def Evaluate_Coordinator_plot(n_round = 5):
    frac_list = []
    for i_round in range(n_round):
        ###### Load model
        i_CO_filename = "RE_CO_" + args.env + "_" + args.learner + "_A" + str(args.agents) + "_C"+ str(args.rounds)+ "_n" + str(args.episodes)  + "_c"+str(i_round+1) + "_a" + str(args.agents)
        i_CO_filename += "_" + args.type + "_" + args.aim + "_s" + str(args.stepsize) + "_m" + str(args.maxiter) + "_r" + str(args.radius) + "_f" + str(args.frac)    
        i_frac = np.loadtxt(args.resdir + i_CO_filename+"_targ.txt")
        print("load from ", args.resdir +  i_CO_filename + ".txt")
        
        frac_list.append(i_frac)
    
    frac_list = np.array(frac_list)
    frac_mean = np.mean(frac_list, axis = 1)
    frac_std = np.std(frac_list, axis = 1)

    fig, ax = plt.subplots()
    x_axis = np.linspace(1,n_round, n_round)
    ax.plot( x_axis,  frac_mean, '*', alpha=0.9, label = "Coordinator: mean")
    
    ax.fill_between(x_axis, frac_mean - frac_std, frac_mean + frac_std, alpha=0.2, color = "green", label = "+- std")
    plt.ylim([0,1])
    plt.xlabel("Communication Rounds (" + str(args.episodes) + " test episodes per round)")
    plt.ylabel("Fraction of Target Actions")
    plt.legend()

    plt.title("Coordinator Performance: "+ str(args.type) + ", " + str(args.learner) +  ", "  + str(args.env) +", " + str(args.agents)+" agents")

    plt.savefig("results/fig_Oct_13/"+"Set1_CO"+".jpg")

       
        

def Evaluate_Agents_plot():
    filename = "RE_"+args.env + "_" + args.learner+ "_A" + str(args.agents) + "_C"+ str(args.rounds) + "_n" + str(args.episodes)  
    for i_round in range(args.rounds):    
        i_targ_list = []
        for i_agent in range(args.agents): # All agents
            if i_agent == 0: # load target action fraction of the malicious agent
                i_filename = filename + "_" + args.type + "_" + args.aim + "_s" + str(args.stepsize) + "_m" + str(args.maxiter) + "_r" + str(args.radius) + "_f" + str(args.frac)+"_c" + str(i_round+1)
            else: # load target action fraction of the clean agents
                i_filename = filename + "_" + "clean" + "_a"+ str(i_agent)+ "_c" + str(i_round+1)
            i_targ_metrix = np.loadtxt(args.resdir + i_filename + "_targ.txt")
            i_targ_list.append(i_targ_metrix)
        
        i_poison_targ_metrix = i_targ_list[0]

        i_clean_targ_list = np.array(i_targ_list[1:])
        i_clean_targ_mean = np.mean(i_clean_targ_list, axis = 0)
        i_clean_targ_std = np.std(i_clean_targ_list, axis = 0)

        i_clean_targ_lb = i_clean_targ_mean - i_clean_targ_std
        i_clean_targ_ub = i_clean_targ_mean + i_clean_targ_std


        fig, ax = plt.subplots()
        x_axis = np.linspace(1,args.episodes, args.episodes)
        ax.plot( x_axis,  i_poison_targ_metrix, '-', color='red', alpha=0.9, label = "Malicous Agent")
        
        ax.plot( x_axis,  i_clean_targ_mean, '-', color='green', alpha=0.9, label = "Mean of "+str(args.agents-1)+" Clean Agents +- std")
        
        ax.fill_between(x_axis, i_clean_targ_lb, i_clean_targ_ub, alpha=0.4, color = "green")
        ax.legend()
        plt.ylim([0,1])
        plt.xlabel("Train Episodes")
        plt.ylabel("Fraction of Target Actions")
        
        

        plt.title("Agent Performance: "+ str(args.type) + ", " + str(args.learner) +  ", "  + str(args.env) +", " + str(args.agents)+" agents, Round " +str(i_round+1) )

        plt.savefig("results/fig_Oct_13/"+"Set1_c"+str(i_round+1)+".jpg")




        
        

if __name__ == '__main__':
    # Poison_Fed_RL()
    Evaluate_Coordinator(n_round = args.rounds, random_seed = 0, attack_flag = 1)
    Evaluate_Coordinator_plot(n_round = args.rounds)
    Evaluate_Agents_plot()
