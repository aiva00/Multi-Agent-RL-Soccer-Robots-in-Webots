import os
import numpy as np
import torch.nn as nn
import torch


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, save_dir,
                 alpha=0.01, beta=0.01, layer1=64,
                 layer2=64, gamma=0.95, t=0.01):
        """
        alpha:
        beta:
        gamma:
        t:
        layer1: neurons in the first fully connected layer
        layer2: neurons in the second fully connected layer
        """

        self.gamma = gamma
        self.t = t
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, layer1, layer2, n_actions,
                                  save_dir=save_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims,
                                    layer1, layer2, n_agents, n_actions,
                                    save_dir=save_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, layer1, layer2, n_actions,
                                         save_dir=save_dir,
                                         name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims,
                                           layer1, layer2, n_agents, n_actions,
                                           save_dir=save_dir,
                                           name=self.agent_name+'_target_critic')

        self.update_network_parameters(t=1)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(
            self.actor.device)
        actions = self.actor.forward(state)

        # Need to add noise for regularization and for optimal solution finding
        noise = torch.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, t=None):
        if t is None:
            t = self.t

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = t*actor_state_dict[name].clone() + \
                (1-t)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = t*critic_state_dict[name].clone() + \
                (1-t)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, layer1_dims, layer2_dims,
                 n_agents, n_actions, name, save_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(save_dir, name)

        self.layer1 = nn.Linear(input_dims+n_agents*n_actions, layer1_dims)
        self.layer2 = nn.Linear(layer1_dims, layer2_dims)
        self.q = nn.Linear(layer2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = torch.nn.functional.relu(
            self.layer1(torch.cat([state, action], dim=1)))
        x = torch.nn.functional.relu(self.layer2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, layer1_dims, layer2_dims,
                 n_actions, name, save_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(save_dir, name)

        self.layer1 = nn.Linear(input_dims, layer1_dims)
        self.layer2 = nn.Linear(layer1_dims, layer2_dims)
        self.pi = nn.Linear(layer2_dims, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = torch.nn.functional.relu(self.layer1(state))
        x = torch.nn.functional.relu(self.layer2(x))
        pi = torch.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple',  alpha=0.01, beta=0.01, layer1=64,
                 layer2=64, gamma=0.99, t=0.01, save_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        save_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     save_dir=save_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_new_states[agent_idx],
                                      dtype=torch.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = torch.tensor(actor_states[agent_idx],
                                     dtype=torch.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat(
            [acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(
                states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma*critic_value_
            critic_loss = torch.nn.functional.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()


class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_counter = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        # We convert everything to bool as an easy way to handle the end state
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):

        index = self.mem_counter % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(
                self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
            actor_new_states, states_, terminal

    def ready(self):
        if self.mem_counter >= self.batch_size:
            return True
