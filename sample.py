from typing import List
import torch
import numpy as np
import pybullet as p
import time
import math
from model import Actor, Agent, Critic
from game import get_all_joint_world_positions, get_joint_limits, \
    apply_rotation, get_all_joint_positions, \
    set_rotation_and_position, get_score
import matplotlib.pyplot as plt


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def compute_reward_to_go(rewards, gamma):
    result = [0]
    for i in reversed(rewards):
        result.append(i+gamma*result[-1])
    print('reward to go' , result, list(reversed(result[1:])))
    return list(reversed(result[1:]))

class Simlater:
    def __init__(self):
        # global lower, diff
        p.connect(p.DIRECT)
        p.setTimeStep(1)
        left = p.loadURDF("models/body.urdf")
        right = p.loadURDF("models/body.urdf")

        lower, upper = get_joint_limits(left)
        diff = upper - lower
        print(upper)
        print(lower)

        self.l, self.r = left, right

        print(torch.tensor(get_all_joint_world_positions(self.l)).flatten())
        set_rotation_and_position(self.l, upper)
        set_rotation_and_position(self.r, lower)

        self.l_last_state = torch.tensor(get_all_joint_world_positions(self.l)).flatten()
        self.r_last_state = torch.tensor(get_all_joint_world_positions(self.r)).flatten()
        #self.l_last_state[0] = lower[0]
        #self.r_last_state[0] = lower[0] + diff[0]
        #set_rotation_and_position(self.l, self.l_last_state)
        #set_rotation_and_position(self.r, self.r_last_state)
        self.diff_tenset = torch.tensor(diff)
        self.lower_tenser = torch.tensor(lower)

        # p.disconnect()
        # set_rotation_and_position(r, [1.1, 1, -2.1, 1, 3, 1, -1, 1, 1, 1,1])
        # set_rotation_and_position(l, [-1.1, 1, -2.1, 1, 3, 1, -1, 1, 1, 1,1])

    def get_answer(self, agent: Agent, amount: int):
        result = []
        states = []
        action = []
        next_states = []
        reward = []
        for i in range(amount):  # Step forward in time
            # while 1:
            #left_action = agent.apply_action(self.l_last_state,self.r_last_state) * self.diff_tenset + self.lower_tenser
            left_action = torch.rand(len(self.diff_tenset)) * self.diff_tenset + self.lower_tenser

            #right_action = torch.rand(len(self.diff_tenset)) * self.diff_tenset + self.lower_tenser
            state = torch.concatenate((self.l_last_state, self.r_last_state),
                                      dim=-1)
            curr_state_score = (get_score(self.l, self.r))
            apply_rotation(self.l, left_action)
            #apply_rotation(self.r, right_action)
            for i in range(10):
                p.stepSimulation()
                # time.sleep(0.1)
            next_state_score = get_score(self.l, self.r)
            self.l_last_state = torch.tensor(get_all_joint_world_positions(self.l)).flatten()
            self.r_last_state = torch.tensor(get_all_joint_world_positions(self.r)).flatten()
            # print(l_last_state)
            next_state = torch.concatenate(
                (self.l_last_state, self.r_last_state), dim=-1)
            l_score, r_score, finshed = get_score(self.l, self.r)
            score = l_score - r_score
            result.append([list(state.detach().numpy()),
                           list(left_action.detach().numpy()),
                           score, list(next_state.detach().numpy())])
            states.append(list(state.detach().numpy()))
            action.append(list(left_action.detach().numpy()))
            next_states.append(list(next_state.detach().numpy()))
            reward.append(score)
            if finshed != 0:
                break

            p.stepSimulation()  # Real time simulation
        return result, states, action, next_states, reward


def sample_batch(simlater: Simlater, agent: Agent, batch_size: int,
                 max_per_run: int = 1000, gamma:float = 0.99):
    rstates, raction, rnext_states, rreward = [], [], [], []
    reward_to_go = []
    for i in range(batch_size):
        print(i)
        data, states, action, next_states, reward = simlater.get_answer(agent,
                                                                        max_per_run)
        rstates, raction, rnext_states, rreward = rstates + states, raction + action, rnext_states + next_states, rreward + reward
        reward_to_go = reward_to_go+compute_reward_to_go(reward,gamma)
    return rstates, raction, rnext_states, rreward,reward_to_go

if __name__ == '__main__':
    simlater = Simlater()
    print(simlater.l_last_state.size(0))
    actor = Actor((simlater.l_last_state.size(0)),
                  (simlater.r_last_state.size(0)),
                  (simlater.diff_tenset.size(0)))
    agent = Agent(actor)
    states, action, next_states, reward,reward_to_go = sample_batch(simlater, agent,
                                                       1)  # simlater.get_answer(agent,1000)
    #print(states, action, next_states, reward, reward_to_go)
    critic = Critic(2 * simlater.l_last_state.size(0))
    gamma = 0.99
    optimizer = torch.optim.Adam(lr=0.001, params=critic.parameters())
    critic.train()
    states = torch.tensor(states)
    action = torch.tensor(action)
    next_states = torch.tensor(next_states)
    reward = torch.tensor(reward)
    l = []
    msk = (reward == 0)
    reward_to_go= torch.tensor(reward_to_go)
    for i in range(10):
        print(i)
        optimizer.zero_grad()
        est_score = critic(states)
        #est_next_score = critic(next_states)
        #td = reward + est_next_score * gamma - est_score*msk
        #loss = torch.sum(td**2)
        loss = torch.sum((est_score-(reward_to_go-reward))**2)
        loss.backward()
        optimizer.step()
        l.append(float(loss))
    printing = list(l)
    draw_buffer = []
    est = []
    for i in range(reward.size(0)):
        est.append(float(critic(states[i])))
        draw_buffer.append(float(
            reward[i] + critic(next_states[i]) * 0.99 * msk[i] - critic(
                states[i])) ** 2)
        #draw_buffer.append(float(critic(next_states[i])-reward_to_go[i])**2)
    print(printing)
    plt.plot(range(len(printing)), printing)
    plt.show()
    print(draw_buffer)
    plt.plot(range(len(draw_buffer)), draw_buffer)
    plt.show()
    print(est)
    plt.plot(range(len(est)), est,'b')
    plt.plot(range(len(reward_to_go)),list(reward_to_go-reward),'g')
    plt.show()
    plt.scatter(est,reward_to_go)
    plt.show()
    #calculated = (
     #       (next_states[:,0]-next_states[:,9])**2+
      #    (next_states[:,1]-next_states[:,10])**2+
       #   (next_states[:,2]-next_states[:,11])**2-
        #  (next_states[:,3]-next_states[:,6])**2-
         # (next_states[:,4]-next_states[:,7])**2-
         # (next_states[:,5]-next_states[:,8])**2)
    #print(-calculated-reward)
    #print(next_states-states)=
    '''
    for i in range(states.size(1)):
        plt.subplot(states.size(1) // 6, 6,i+1)
        plt.scatter(states[:,i],est)
        plt.scatter(states[:, i], reward_to_go)
    plt.show()
    '''
