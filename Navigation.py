from unityagents import UnityEnvironment
import numpy as np
from collections import deque
#from ddpq_agent import Agent
import matplotlib.pyplot as plt
import torch

from ddpg_agent import Agent

class Environemnt:
    def __init__(self, env_path, train_mode=True):

        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.agents = self.env_info.agents
        self.train_mode = train_mode


    def get_number_of_agents(self):
        return  len(self.agents)

    def get_number_of_actions(self):
        return  self.brain.vector_action_space_size

    def get_current_state(self):
       return self.env_info.vector_observations

    def reset_environment(self):
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self.env_info

    def make_action(self, action_id):
        self.env_info = self.env.step(action_id)[self.brain_name]
        return self.env_info

    def is_done(self):
        return self.env_info.local_done

    def get_current_reward(self):
        reward = self.env_info.rewards
        return reward

    def close(self):
        self.env.close()

def play_env(env, agent, num_agents=2, trials=300, max_t=1000):
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))


    for i in range(trials):
        print('Start Trial...')
        env.reset_environment()
        state = env.get_current_state()
        agent.reset()
        scores = np.zeros(num_agents)

        for j in range(max_t):
            actions = agent.act(state)
            env_info = env.make_action(actions)
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            state = next_state
            scores+=reward

            if np.any(done):
                print('Done.')
                break

        print("Current Reward:", np.mean(scores))
    env.close()


def play():
    environment = Environemnt(env_path="envs/3/Tennis.exe",
                              train_mode=False)

    num_agents = environment.get_number_of_agents()
    state_size = len(environment.get_current_state()[0])
    action_size = environment.get_number_of_actions()


    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  random_seed=0,
                  num_agents = num_agents)
    play_env(environment, agent, num_agents)

def ddpg(agent, environment,  num_agents, n_episodes=1000, max_t=1000, print_every=100, goal=0.5):
    scores_window = deque(maxlen=100)
    scores_episode = []

    for i_episode in range(1, n_episodes + 1):
        environment.reset_environment()
        state = environment.get_current_state()
        # for agent in agents:
        #     agent.reset()
        agent.reset()
        scores = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(state)
            environment.make_action(actions)
            next_state = environment.get_current_state()
            reward = environment.get_current_reward()
            done = environment.is_done()
            #raise Exception
            agent.step(t, state, actions, reward, next_state, done)

            state = next_state
            scores += reward
            if t % 20:
                print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                      .format(t, np.mean(scores), np.min(scores), np.max(scores)), end="")
            if np.any(done):
                break
        score = np.max(scores)
        scores_window.append(score)  # save most recent score
        scores_episode.append(score)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)),
              end="\n")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return scores_episode, scores_window

def train():

    environment = Environemnt(env_path="envs/3/Tennis.exe")

    num_agents = environment.get_number_of_agents()
    state_size = len(environment.get_current_state()[0])
    action_size = environment.get_number_of_actions()

    agents = []


    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  random_seed=0,
                  num_agents = num_agents
                )

    scores, scores_window = ddpg(agent, environment, num_agents, n_episodes=5000)

    np.save('ddpg' + '_scores_new.npy', np.array(scores))
    np.save('ddpg' + '_scores_window.npy', np.array(scores_window))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    train()