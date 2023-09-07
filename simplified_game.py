from typing import List
import torch
import numpy as np
import pybullet as p
import time
import math

from matplotlib import pyplot as plt

from model import Critic, Agent, Actor

static = -1
x_corr = 0
y_corr = 1
body = 2
shoulder_y = 3
shoulder_z = 4
larger_arm = 5
elbow = 6
smaller_arm = 7
wrist_z = 8
wrist_y = 9
sward = 10
sward_top = 11

state_points = [body,shoulder_y,elbow,wrist_y,sward_top]
def compute_reward_to_go(rewards, gamma):
    result = [0]
    for i in reversed(rewards):
        result.append(i + gamma * result[-1])
    #print('reward to go', result, list(reversed(result[1:])))
    return list(reversed(result[1:]))

def get_joint_limits(robot_id,client):

    action_to_joint_index = []
    num_joints = p.getNumJoints(robot_id,physicsClientId=client)

    # initialize a list to store joint positions
    lower, upper = [], []

    # iterate over all joints
    for joint_id in range(num_joints):
        # get the joint state
        joint_info = p.getJointInfo(robot_id, joint_id,physicsClientId=client)
        if not joint_info[2]==p.JOINT_FIXED:
            action_to_joint_index.append(joint_info[0])
            print(joint_info)
        # append the joint position to the list
            if joint_info[8]<joint_info[9]:
                lower.append(joint_info[8])
                upper.append(joint_info[9])
            else:
                lower.append(-3.14)
                upper.append(3.14)
    return np.asarray(lower), np.asarray(upper), action_to_joint_index

class Simplified_game:
    def __init__(self, agent, gui=False,drop_out_rate = 0.5,):
        if gui:
            self.client=p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        self.l = p.loadURDF('models/body.urdf',physicsClientId=self.client,flags=p.URDF_USE_SELF_COLLISION)

        p.resetJointState(self.l, 0, -1.5,physicsClientId=self.client)
        p.resetJointState(self.l, 1, 0,physicsClientId=self.client)
        p.resetJointState(self.l, 2, 0,physicsClientId=self.client)
        self.r = p.loadURDF('models/body.urdf',physicsClientId=self.client,flags=p.URDF_USE_SELF_COLLISION)
        p.resetJointState(self.r, 0, 1.5,physicsClientId=self.client)
        p.resetJointState(self.r, 1, 0,physicsClientId=self.client)
        p.resetJointState(self.r, 2, 3.14,physicsClientId=self.client)
        self.agent = agent

        self.l_lower, self.l_upper, self.l_index = get_joint_limits(self.l,self.client)
        self.l_diff = self.l_upper-self.l_lower
        self.r_lower, self.r_upper, self.r_index = get_joint_limits(self.r,
                                                                    self.client)
        self.r_diff = self.r_upper-self.r_lower
        self.drop_out_rate=drop_out_rate

    def get_joint_limits(robot_id):

        num_joints = p.getNumJoints(robot_id)

        # initialize a list to store joint positions
        lower, upper = [], []

        # iterate over all joints
        for joint_id in range(num_joints):
            # get the joint state
            joint_info = p.getJointInfo(robot_id, joint_id)

            print(joint_info)
            # append the joint position to the list
            if joint_info[8] < joint_info[9]:
                lower.append(joint_info[8])
                upper.append(joint_info[9])
            else:
                lower.append(-3.14)
                upper.append(3.14)
        return np.asarray(lower), np.asarray(upper)

    def get_left_state(self):
        result = ()
        for i in state_points:
            result = result+p.getLinkState(self.l, i,physicsClientId=self.client)[0]+ \
                     p.getLinkState(self.r, i, physicsClientId=self.client)[0]
        return torch.tensor(result)

        #return np.concatenate((self.left_point+0, self.right_point+0))  # ,self.right_point
    def get_right_state(self):
        result = ()
        for i in state_points:
            result = result + \
                     p.getLinkState(self.r, i, physicsClientId=self.client)[0] + \
                     p.getLinkState(self.l, i, physicsClientId=self.client)[0]
        return torch.tensor(result)

    def get_score(self):
        #return -np.sum((self.left_point-self.right_point) ** 2)
        contacts = p.getContactPoints(bodyA=self.l, bodyB=self.r)
        a, b = 0, 0
        for i in contacts:
            if i[3] == body and i[4] in [sward_top,sward]:
                a = 1
            if i[4] == body and i[3] in [sward_top,sward]:
                b = 1
        return np.sum((np.asarray(
            p.getLinkState(self.l, body,physicsClientId=self.client)[0]) - np.asarray(
            p.getLinkState(self.r, sward_top,physicsClientId=self.client)[0])) ** 2)-np.sum((np.asarray(
            p.getLinkState(self.r, body,physicsClientId=self.client)[0]) - np.asarray(
            p.getLinkState(self.l, sward_top,physicsClientId=self.client)[0])) ** 2)+(b-a)*50

    def apply(self, l_action, r_action):
        # x,y, rotation
        #print('b',self.get_state())
        #print('l',action)
        #self.left_point += l_action
        #self.right_point+=r_action
        #print('a',self.get_state())
        #return
        #print(l_action,self.l_diff)
        l_action=l_action.detach()*self.l_diff+self.l_lower
        r_action = r_action.detach()*self.r_diff+self.r_lower
        for i, j in zip(self.l_index,range(len(l_action))):
            p.setJointMotorControl2(self.l, i, p.POSITION_CONTROL,
                                l_action[j], force=100,physicsClientId=self.client)
        for i, j in zip(self.r_index,range(len(r_action))):
            p.setJointMotorControl2(self.r, i, p.POSITION_CONTROL,
                                r_action[j], force=100,physicsClientId=self.client)
        #p.setJointMotorControl2(self.l, 1, p.POSITION_CONTROL,
         #                       l_action[1]*4, force=100,physicsClientId=self.client)
        #p.setJointMotorControl2(self.l, 2, p.POSITION_CONTROL,
        #                        l_action[2]*6.28, force=100,physicsClientId=self.client)

        #p.setJointMotorControl2(self.r, 0, p.POSITION_CONTROL,
         #                       r_action[0]*4, force=100,physicsClientId=self.client)
        #p.setJointMotorControl2(self.r, 1, p.POSITION_CONTROL,
         #                       r_action[1]*4, force=100,physicsClientId=self.client)
        #p.setJointMotorControl2(self.r, 2, p.POSITION_CONTROL,
         #                       r_action[2]*6.28, force=100,physicsClientId=self.client)
        for i in range(10):
            p.stepSimulation()
            #time.sleep(0.1)


    def get_answer(self, amount: int):
        result = []
        states = []
        action = []
        next_states = []
        reward = []
        log_prob = []
        for i in range(amount):  # Step forward in time
            state, left_prob, left_action, right_prob, right_action, next_state, score = self.step()
            result.append([list(state),
                           list(left_action),
                           score, list(next_state)])
            states.append(list(state))
            action.append(left_action)
            next_states.append(list(next_state))
            reward.append(score)
            log_prob.append(left_prob)
            # if finshed != 0:
            #   break

            # p.stepSimulation()  # Real time simulation
        return result, states, action, next_states, reward, log_prob

    def step(self):
        state = self.get_left_state()  # torch.tensor((get_state(self.l)+get_state(self.r)))
        left_prob, left_action = self.agent.apply_action(state)
        right_prob, right_action = self.agent.apply_action(
            self.get_right_state())
        msk = torch.rand(right_action.size(0))>self.drop_out_rate
        right_action = right_action*msk+torch.rand(right_action.size(0))*(~msk)
        self.apply(left_action, right_action)
        next_state = self.get_left_state()
        score = self.get_score()
        return state, left_prob, left_action, right_prob, right_action, next_state,score


class Similater:
    def __init__(self, batch_size:int, agent:Agent):
        self.similations = []
        for i in range(batch_size):
            self.similations.append(Simplified_game(agent))
    def step(self, num_step=1):
        states = []
        action = []
        next_states = []
        reward = []
        log_prob = []
        for samilator in self.similations:  # Step forward in time
            for _ in range(num_step):
                state, left_prob, left_action, right_prob, right_action, next_state,\
                score = samilator.step()
                states.append(list(state))
                action.append(left_action)
                next_states.append(list(next_state))
                reward.append(score)
                log_prob.append(left_prob)
            # if finshed != 0:
            #   break

            # p.stepSimulation()  # Real time simulation
        return torch.tensor(states), action, torch.tensor(next_states),\
               torch.tensor(reward), torch.concatenate(log_prob)
def sample_batch(simlater: Simplified_game, batch_size: int,
                 max_per_run: int = 1000, gamma: float = 0.99):
    rstates, raction, rnext_states, rreward, rlog_prob = [], [], [], [], []
    reward_to_go = []
    for i in range(batch_size):
        #print(i)
        data, states, action, next_states, reward, log_prob = simlater.get_answer(
            max_per_run)
        #print('d', torch.tensor(states) - torch.tensor(next_states))
        rstates, raction, rnext_states, rreward, rlog_prob =\
            rstates + states, raction + action, rnext_states + next_states,\
            rreward + reward, rlog_prob+log_prob
        reward_to_go = reward_to_go + compute_reward_to_go(reward, gamma)
    #print('rlp',torch.tensor(rlog_prob))
    return torch.tensor(rstates, dtype=torch.float32), torch.concatenate(raction),\
           torch.tensor(rnext_states, dtype=torch.float32), torch.tensor(rreward),\
           reward_to_go, torch.concatenate(rlog_prob)


def get_state(pid):
    # for i in range(p.getNumJoints(pid)):
    #   print(p.getJointInfo(pid,i))
    return p.getLinkState(pid, sward_top)[0] + p.getLinkState(pid, body)[0]

def train(actor:torch.nn.Module, critic:torch.nn.Module, gamma:float,s:Similater,epoch:int,
          critic_updates:int, lr:float = 0.1,normizalation_factor:float = 0.0001):
    critic_optimizer = torch.optim.Adam(lr=lr, params=critic.parameters())
    actor_optimizer = torch.optim.Adam(lr=lr, params=actor.parameters())
    critic.train()
    actor.train()
    l = []
    a = []
    for i in range(epoch):
        states, action, next_states, reward, log_porb = s.step(10)
        # msk = (reward == 0)
        if i % (epoch // 100 + 1) == 0:
            print(int(i * 100 / epoch), '%')
        td = 0
        for i in range(critic_updates):
            critic_optimizer.zero_grad()
            est_score = critic(states)
            est_next_score = critic(next_states)
            td = reward + est_next_score.T * gamma - est_score.T
            loss = torch.sum(td ** 2 + normizalation_factor * (
                        est_next_score.T ** 2 + est_next_score.T ** 2))
            loss.backward()
            critic_optimizer.step()
            l.append(float(loss))
        # print(est_score.T - reward_to_go)
        # print(log_porb.size())
        # print(td.size())
        # print(action.size())
        actor_loss = -torch.sum(log_porb * td.detach())
        actor_loss.backward()
        actor_optimizer.step()
        a.append(float(actor_loss))
    return l, a

def train_and_save(state_length,action_length,hidden ):
    critic = torch.nn.Sequential(
        torch.nn.Linear(state_length, hidden, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 2 * hidden, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * hidden, 1))  # Critic(2 * s.l_last_state.size(0))
    gamma = 0.99

    actor = Actor(state_length, hidden, action_length)
    agent = Agent(actor)

    s = Similater(10, agent)

    epoch = 100
    critic_updates = 10
    normizalation_factor = 0.0001
    # print(action)
    l, a = train(actor, critic, gamma, s, epoch, critic_updates,
                 normizalation_factor=normizalation_factor)

    torch.save(actor.state_dict(), 'model_dicts/actor.pth')
    torch.save(critic.state_dict(), 'model_dicts/critic.pth')
    printing = list(l)
    draw_buffer = []
    est = []
    # for i in range(reward.size(0)):
    #   est.append(float(critic(states[i])))

    # draw_buffer.append(float(critic(next_states[i])-reward_to_go[i])**2)
    print(printing)
    plt.plot(printing)
    plt.show()
    print(a)
    plt.plot(a)
    plt.show()
def play(state_length, action_length, hidden):
    actor = Actor(state_length, hidden, action_length)
    actor.load_state_dict(torch.load('model_dicts/actor.pth'))
    agent = Agent(actor)
    s = Simplified_game(agent, True, 0)
    while 1:
        p.stepSimulation()
        #s.step()
        time.sleep(0.01)
if __name__ == '__main__':

    state_length = len(state_points)*6
    action_length = 10
    hidden = 120
    #train_and_save(state_length, action_length, hidden)
    play(state_length, action_length, hidden)
    #cid = p.connect(p.GUI)
    #a = p.loadURDF('models/simplified.urdf')
    #for i in range(p.getNumJoints(a)):
     #   print(p.getBodyInfo(a,i))
        #print(p.getJointState(a, i))
     #   print(p.getJointInfo(a,i))
      #  print(p.getLinkState(a,i))
    #print(get_joint_limits(a,cid))
    #while 1:
     #   p.stepSimulation()

       # time.sleep(0.01)

    '''
    print(draw_buffer)
    plt.plot(range(len(draw_buffer)), draw_buffer)
    plt.show()
    print(est)
    plt.plot(range(len(est)), est, 'b')
    plt.plot(range(len(reward_to_go)), list(reward_to_go), 'g')
    plt.show()
    plt.plot(reward_to_go)
    plt.show()
    print(reward_to_go)
    for i in range(states.size(1)):
        indexes = torch.argsort(states[:,i])
        plt.plot(states[:,i][indexes], reward_to_go[indexes])
        plt.plot(states[:,i][indexes], np.asarray(est)[indexes])
        plt.show()
    print((reward_to_go), critic(states), states)
    print(states, next_states)
    #plt.plot((reward+gamma*critic(next_states).detach().numpy().T-critic(states).detach().numpy().T))
    #plt.plot(reward_to_go-critic(states).detach().numpy().T)
    #plt.show()
    # plt.scatter(est, reward_to_go)
    # plt.show()
    # calculated = (
    #       (next_states[:,0]-next_states[:,9])**2+
    #    (next_states[:,1]-next_states[:,10])**2+
    #   (next_states[:,2]-next_states[:,11])**2-
    #  (next_states[:,3]-next_states[:,6])**2-
    # (next_states[:,4]-next_states[:,7])**2-
    # (next_states[:,5]-next_states[:,8])**2)
    # print(-calculated-reward)
    # print(next_states-states)=
    '''
    '''
    for i in range(states.size(1)):
        plt.subplot(states.size(1) // 6, 6,i+1)
        plt.scatter(states[:,i],est)
        plt.scatter(states[:, i], reward_to_go)
    plt.show()
    '''
