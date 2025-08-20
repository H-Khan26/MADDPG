import torch.nn as nn
import torch
import random
import pickle
from pettingzoo.mpe import simple_tag_v3


class AInet:
    # n_agent_types is the amount of unique agent types
    # agent_types is a list that has one element for each agent, with int values 0 to n_agent_types denoting which type each is
    # other_sub_sizes is a list within a list, each with length == n_agent_types, for the subset size within the observation of [observer_type][observed_type]
    # slfgbl_sub_sizes is a list that has length == n_agent_types, each being the slfgbl_sub_size within that an agent of that type's observation
    # action_sub_sizes is a list that has length == n_agent_types, each being the action_sub_size that an agent of that type contributes to the collective action
    def __init__(self, n_agent_types, agent_types, other_sub_sizes, slfgbl_sub_sizes, action_sub_sizes, lr=0.005, device="cpu"):
        self.n_agents = len(agent_types)
        self.agent_types = agent_types
        self.social_modules = []
        self.self_modules = []
        for observer_type in range(n_agent_types):
            social_module_set = []
            for observed_type in range(n_agent_types):
                social_module_set.append(Awareness(3*(other_sub_sizes[observer_type][observed_type]+slfgbl_sub_sizes[observer_type]), action_sub_sizes[observed_type], lr).to(device))
            self.social_modules.append(social_module_set)
            self.self_modules.append(Awareness(3*slfgbl_sub_sizes[observer_type], action_sub_sizes[observer_type], lr).to(device))

    # sub_assigns is a vector with length equal to len(obs) == (n_total-1)*other_sub_size + slfgbl_sub_size
    #  defining how it's broken down, -1 for things that are always part of slfgbl, else it's the index of action order
    def forward(self, sub_assigns, observer_ind, curr_observation, last_observation):
        observer_type = self.agent_types[observer_ind]
        slfgbl_assigns = sub_assigns == -1
        curr_slfgbl = curr_observation[slfgbl_assigns]
        last_slfgbl = last_observation[slfgbl_assigns]
        out_subs = []
        for observed_ind in range(self.n_agents):
            if observed_ind == observer_ind:
                out_subs.append(self.self_modules[observer_type](torch.cat([curr_slfgbl, last_slfgbl, curr_slfgbl - last_slfgbl])))
            else:
                other_assigns = sub_assigns == observed_ind
                curr_bundle = torch.cat([curr_observation[other_assigns], curr_slfgbl])
                last_bundle = torch.cat([last_observation[other_assigns], last_slfgbl])
                out_subs.append(self.social_modules[observer_type][self.agent_types[observed_ind]](torch.cat([curr_bundle, last_bundle, curr_bundle - last_bundle])))
        torch.cat(out_subs)


class Awareness(nn.Module):
    def __init__(self, bundle_size, action_sub_size, lr=0.005):
        super(Awareness, self).__init__()
        self.fc1 = nn.Linear(bundle_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_sub_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, bundle: torch.Tensor):
        x = torch.relu(self.fc1(bundle))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.005)
        self.optimizer.step()


def one_hot(index, size):
    vec = torch.zeros(size)
    vec[index] = 1.0
    return vec


def pre_train_ai_net(episode_count, max_episode_length, percent_to_train_on, lr=0.0001, device="cpu"):
    env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=max_episode_length)
    observations, infos = env.reset()
    agents = env.agents
    n_agent_types = 2
    agent_types = [0, 0, 0, 1]
    other_sub_size = [[2, 4], [2, 0]]
    slfgbl_sub_sizes = [8, 8]
    action_sub_sizes = [5, 5]
    sub_assignments = {0: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 2, 2, 3, 3, 3, 3]),
                       1: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 2, 2, 3, 3, 3, 3]),
                       2: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 3, 3, 3, 3]),
                       3: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 2, 2])}
    ai_net = AInet(n_agent_types, agent_types, other_sub_size, slfgbl_sub_sizes, action_sub_sizes, lr, device)
    for i in range(episode_count):
        observations, infos = env.reset()
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            last_observations = observations
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for observer_ind in range(len(agents)):
                observer_type = agent_types[observer_ind]
                slfgbl_assigns = sub_assignments[observer_ind] == -1
                curr_slfgbl = torch.tensor(observations[agents[observer_ind]])[slfgbl_assigns]
                last_slfgbl = torch.tensor(last_observations[agents[observer_ind]])[slfgbl_assigns]
                for observed_ind in range(len(agents)):
                    if random.uniform(0, 1) > percent_to_train_on:
                        continue
                    if observed_ind == observer_ind:
                        a_pred = ai_net.self_modules[observer_type](
                            torch.cat([curr_slfgbl, last_slfgbl, curr_slfgbl - last_slfgbl]))
                        loss = nn.MSELoss()(one_hot(actions[agents[observer_ind]], 5), a_pred)
                        print("Self-Awareness Module [Agent Type " + str(agent_types[observer_ind]) + "], MSE:: " + str(loss.item()))
                        ai_net.self_modules[observer_type].update(loss)
                    else:
                        other_assigns = sub_assignments[observer_ind] == observed_ind
                        curr_bundle = torch.cat([torch.tensor(observations[agents[observer_ind]])[other_assigns], curr_slfgbl])
                        last_bundle = torch.cat([torch.tensor(last_observations[agents[observer_ind]])[other_assigns], last_slfgbl])
                        a_pred = ai_net.social_modules[observer_type][agent_types[observed_ind]](torch.cat([curr_bundle, last_bundle, curr_bundle - last_bundle]))
                        loss = nn.MSELoss()(one_hot(actions[agents[observed_ind]], 5), a_pred)
                        print("Social-Awareness Module [Agent Type " + str(agent_types[observer_ind]) + ", Observing Type " + str(agent_types[observed_ind]) + "], MSE:: " + str(loss.item()))
                        ai_net.social_modules[observer_type][agent_types[observed_ind]].update(loss)
    env.close()
    return ai_net


if __name__ == "__main__":
    model = pre_train_ai_net(200, 20, 0.8, 0.001, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with open('AI_Net.pkl', 'wb') as f:
        pickle.dump(model, f)