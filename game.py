import copy
import random
from typing import List

import numpy
import torch
import numpy as np
import pybullet as p
import time
import math

from matplotlib import pyplot as plt

from model import Critic, Agent, Actor, Actor_critic

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
right_sward = 19
right_sward_top = 20

state_points = [body, shoulder_y, elbow, wrist_y, sward_top, 13, 15, 18, 20]
adj_matrix = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1, 1]
                           ])


def facing_loss(x0, y0, x1, y1, angle):
    norm = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5
    if norm == 0:
        return 0
    return ((y1 - y0) * np.sin(angle) + (x1 - x0) * np.cos(angle)) / norm


def shortest_distence(x_0, x_1, y_0, y_1):
    # print('prem', x_0,x_1,y_0,y_1)
    e_x = x_1 - x_0
    e_y = y_1 - y_0
    n = np.cross(e_x, e_y)
    if np.sum(n ** 2) == 0:
        return np.sum(np.cross(e_x, (y_0 - x_0)) ** 2)
    t = np.linalg.solve(np.asarray([e_y, e_x, n]).T, x_0 - y_0)
    t2 = t[0]
    t1 = -t[1]
    px = x_0 + t1 * e_x
    py = y_0 + t2 * e_y
    if t1 < 0:
        px = x_0
    elif t1 > 1:
        px = x_1
    if t2 < 0:
        py = y_0
    elif t2 > 1:
        py = y_1

    # print(px,py)
    return np.sum((px - py) ** 2)


def compute_reward_to_go(rewards, gamma):
    result = [0]
    for i in reversed(rewards):
        result.append(i + gamma * result[-1])
    # print('reward to go', result, list(reversed(result[1:])))
    return list(reversed(result[1:]))


def get_joint_limits(robot_id, client):

    action_to_joint_index = []
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)

    # initialize a list to store joint positions
    lower, upper = [], []

    # iterate over all joints
    for joint_id in range(num_joints):
        # get the joint state
        joint_info = p.getJointInfo(robot_id, joint_id, physicsClientId=client)
        # print(joint_info)
        if not joint_info[2] == p.JOINT_FIXED:
            action_to_joint_index.append(joint_info[0])
            # append the joint position to the list
            if joint_info[8] < joint_info[9]:
                lower.append(joint_info[8])
                upper.append(joint_info[9])
            else:
                lower.append(-3.14)
                upper.append(3.14)
    return np.asarray(lower), np.asarray(upper), action_to_joint_index


def get_curr_state(robot_id, client):
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)
    result = []
    for joint_id in range(num_joints):
        result.append(
            p.getJointState(robot_id, joint_id, physicsClientId=client)[0])
    return result


def set_curr_joint(robot_id, values, client):
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)
    for i in range(num_joints):
        p.resetJointState(robot_id, i, values[i], physicsClientId=client)


def get_random_position(x_max, y_max):
    return (np.random.random() - 1) * 2 * x_max, (
            np.random.random() - 1) * 2 * y_max


class Game:
    def __init__(self, agent, gui=False, drop_out_rate=0.5):
        self.gui = gui
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        self.l = p.loadURDF('models/two_arms.urdf', physicsClientId=self.client,
                            flags=p.URDF_USE_SELF_COLLISION)

        self.r = p.loadURDF('models/two_arms.urdf', physicsClientId=self.client,
                            flags=p.URDF_USE_SELF_COLLISION)

        self.reset()
        '''
        p.resetJointState(self.l, 2, 0, physicsClientId=self.client)
        p.resetJointState(self.l, 4, 3.14 / 2, physicsClientId=self.client)
        p.resetJointState(self.l, 6, -3.14 / 2, physicsClientId=self.client)
        # print(p.getJointInfo(self.l,6,physicsClientId=self.client))
        p.resetJointState(self.l, 13, 3.14 / 2, physicsClientId=self.client)
        p.resetJointState(self.l, 15, -3.14 / 2, physicsClientId=self.client)
        p.resetJointState(self.r, 2, 3.14, physicsClientId=self.client)
        p.resetJointState(self.r, 4, 3.14 / 2, physicsClientId=self.client)
        p.resetJointState(self.r, 6, -3.14 / 2, physicsClientId=self.client)
        # print(p.getJointInfo(self.l,6,physicsClientId=self.client))
        p.resetJointState(self.r, 13, 3.14 / 2, physicsClientId=self.client)
        p.resetJointState(self.r, 15, -3.14 / 2, physicsClientId=self.client)
        '''
        self.agent = agent

        self.l_lower, self.l_upper, self.l_index = get_joint_limits(self.l,
                                                                    self.client)
        self.l_diff = self.l_upper - self.l_lower
        self.r_lower, self.r_upper, self.r_index = get_joint_limits(self.r,
                                                                    self.client)
        self.r_diff = self.r_upper - self.r_lower
        self.drop_out_rate = drop_out_rate

        # self.l_starting_state = get_curr_state(self.l, self.client)
        # self.r_starting_state = get_curr_state(self.r, self.client)

    def reset(self):
        x_pos, y_pos = get_random_position(3, 3)
        p.resetJointState(self.l, 0, x_pos, physicsClientId=self.client)
        p.resetJointState(self.l, 1, y_pos, physicsClientId=self.client)
        p.resetJointState(self.l, body, (np.random.rand() - 0.5) * 3.14 * 2,
                          physicsClientId=self.client)
        # print(p.getJointInfo(self.l, 15, physicsClientId=self.client))

        x_pos, y_pos = get_random_position(1.5, 1.5)
        p.resetJointState(self.r, 0, x_pos, physicsClientId=self.client)
        p.resetJointState(self.r, 1, y_pos, physicsClientId=self.client)
        p.resetJointState(self.r, body, (np.random.rand() - 0.5) * 3.14 * 2,
                          physicsClientId=self.client)

    def get_left_joint_pos(self):
        result = []
        for i in state_points:
            result = result + [
                p.getLinkState(self.l, i, physicsClientId=self.client)[0]]
        return torch.tensor(result)

    def get_right_joint_pos(self):
        result = []
        for i in state_points:
            result = result + [
                p.getLinkState(self.r, i, physicsClientId=self.client)[0]]
        return torch.tensor(result)

    def get_left_state(self):
        result = []
        for i in state_points:
            result = result + \
                     [p.getLinkState(self.r, i, physicsClientId=self.client)[0],
                      p.getLinkState(self.l, i, physicsClientId=self.client)[0]]
        res = torch.tensor(result) - torch.tensor(
            p.getLinkState(self.l, body, physicsClientId=self.client)[0])
        return res.flatten()[3:]
        # return np.concatenate((self.left_point+0, self.right_point+0))  # ,self.right_point

    def get_right_state(self):
        result = []
        for i in state_points:
            result = result + \
                     [p.getLinkState(self.r, i, physicsClientId=self.client)[0],
                      p.getLinkState(self.l, i, physicsClientId=self.client)[0]]
        res = torch.tensor(result) - torch.tensor(
            p.getLinkState(self.r, body, physicsClientId=self.client)[0])
        return res.flatten()[3:]

    def get_contect(self):
        contacts = p.getContactPoints(bodyA=self.l, bodyB=self.r)
        a, b = 0, 0
        for i in contacts:
            if i[3] == body and i[4] in [sward_top, sward, right_sward,
                                         right_sward_top]:
                # print('a')
                a = 1
            if i[4] == body and i[3] in [sward_top, sward, right_sward,
                                         right_sward_top]:
                # print('b')
                b = 1
        return a, b

    def get_score(self):
        # return -np.sum((self.left_point-self.right_point) ** 2)

        body_dis_factor = -1
        attack_ratio = 0.8
        hit_factor = 50
        facing_loss_factor = 9
        contacts = p.getContactPoints(bodyA=self.l, bodyB=self.r)
        a, b = self.get_contect()
        # distance between l body and r sward, minimize, -
        self_body_bottom = np.asarray(
            p.getLinkState(self.l, body, physicsClientId=self.client)[0])
        self_body_top = np.asarray(
            p.getLinkState(self.l, body, physicsClientId=self.client)[0])
        self_body_top[2] = 1.5
        # self_body_top = (self_body_top[0],self_body_top[1],1.5)
        enemy_body_bottom = \
            np.asarray(
                p.getLinkState(self.r, body, physicsClientId=self.client)[0])
        enemy_body_top = \
            np.asarray(
                p.getLinkState(self.r, body, physicsClientId=self.client)[0])
        enemy_body_top[2] = 1.5
        # enemy_body_top = (enemy_body_top[0],enemy_body_top[1],1.5)

        self_l_sward_bottom = np.asarray(
            p.getLinkState(self.l, sward_top - 1, physicsClientId=self.client)[
                0])
        self_l_sward_top = np.asarray(
            p.getLinkState(self.l, sward_top, physicsClientId=self.client)[0])
        self_r_sward_bottom = np.asarray(
            p.getLinkState(self.l, right_sward_top - 1,
                           physicsClientId=self.client)[0])
        self_r_sward_top = np.asarray(p.getLinkState(self.l, right_sward_top,
                                                     physicsClientId=self.client)[
                                          0])

        enemy_l_sward_bottom = \
            np.asarray(p.getLinkState(self.r, sward_top - 1,
                                      physicsClientId=self.client)[0])
        enemy_l_sward_top = \
            np.asarray(
                p.getLinkState(self.r, sward_top, physicsClientId=self.client)[
                    0])
        enemy_r_sward_bottom = np.asarray(
            p.getLinkState(self.r, right_sward_top - 1,
                           physicsClientId=self.client)[0])
        enemy_r_sward_top = \
            np.asarray(p.getLinkState(self.r, right_sward_top,
                                      physicsClientId=self.client)[0])

        self_l_to_enemy = shortest_distence(self_l_sward_bottom,
                                            self_l_sward_top, enemy_body_bottom,
                                            enemy_body_top)
        self_r_to_enemy = shortest_distence(self_r_sward_bottom,
                                            self_r_sward_top, enemy_body_bottom,
                                            enemy_body_top)

        enemy_l_to_self = shortest_distence(enemy_l_sward_bottom,
                                            enemy_l_sward_top, self_body_bottom,
                                            self_body_top)
        enemy_r_to_self = shortest_distence(enemy_r_sward_bottom,
                                            enemy_r_sward_top, self_body_bottom,
                                            self_body_top)

        shortest_enemy_sward_to_self = np.exp(
            -10 * (enemy_r_to_self ** 2)) + np.exp(-10 * (
                enemy_l_to_self ** 2))  # min(enemy_r_to_self,enemy_l_to_self)
        shortest_self_sward_to_enemy = np.exp(
            -10 * (self_l_to_enemy ** 2)) + np.exp(-10 * ((
                                                              self_r_to_enemy) ** 2))  # min(self_l_to_enemy, self_r_to_enemy)
        '''
        self_ldis = np.sum((np.asarray(
            p.getLinkState(self.l, body, physicsClientId=self.client)[
                0]) - np.asarray(
            p.getLinkState(self.r, sward_top, physicsClientId=self.client)[
                0])) ** 2)
        # distance between r body and l sward, maximize, +
        enemy_ldis = np.sum((np.asarray(
            p.getLinkState(self.r, body, physicsClientId=self.client)[
                0]) - np.asarray(
            p.getLinkState(self.l, sward_top, physicsClientId=self.client)[
                0])) ** 2)
        self_rdis = np.sum((np.asarray(
            p.getLinkState(self.l, body, physicsClientId=self.client)[
                0]) - np.asarray(
            p.getLinkState(self.r, right_sward_top,
                           physicsClientId=self.client)[0])) ** 2)
        enemy_rdis = np.sum((np.asarray(
            p.getLinkState(self.r, body, physicsClientId=self.client)[
                0]) - np.asarray(
            p.getLinkState(self.l, right_sward_top,
                           physicsClientId=self.client)[0])) ** 2)
        bod_dis = np.sum((np.asarray(
            p.getLinkState(self.r, body, physicsClientId=self.client)[
                0]) - np.asarray(
            p.getLinkState(self.l, body,
                           physicsClientId=self.client)[0])) ** 2)
       
        
        return -min(enemy_ldis, enemy_rdis) + attack_ratio * min(self_ldis,
                                                                 self_rdis) + body_dis_factor * bod_dis + hit_factor * (
                       b - a), attack_ratio * min(enemy_ldis, enemy_rdis) - min(
            self_ldis,
            self_rdis) + body_dis_factor * bod_dis + hit_factor * (
                       a - b), b - a
         '''
        l_body_pos = p.getLinkState(self.l, body,
                                    physicsClientId=self.client)[0]
        r_body_pos = p.getLinkState(self.r, body,
                                    physicsClientId=self.client)[0]
        bod_dis = np.sum((np.asarray(l_body_pos) - np.asarray(r_body_pos) ** 2))

        l_facing_loss = facing_loss(l_body_pos[0], l_body_pos[1], r_body_pos[0],
                                    r_body_pos[1], p.getJointState(self.l, body,
                                                                   physicsClientId=self.client)[
                                        0]) - 1
        r_facing_loss = facing_loss(r_body_pos[0], r_body_pos[1], l_body_pos[0],
                                    l_body_pos[1], p.getJointState(self.r, body,
                                                                   physicsClientId=self.client)[
                                        0]) - 1
        # print('a',l_facing_loss,r_facing_loss)
        if self.gui:
            print(bod_dis)
        # bod_dis=np.exp(-10*((np.sqrt(bod_dis)-2))**2)-1
        if self.gui:
            print('b', bod_dis)
        self_normal = (-shortest_enemy_sward_to_self + attack_ratio *
                       shortest_self_sward_to_enemy) - 2
        body_dis = body_dis_factor * bod_dis
        return 0.1 * (
                self_normal + body_dis + facing_loss_factor * l_facing_loss) + hit_factor * (
                       b - a), \
               0.1 * (
                       -self_normal + body_dis + facing_loss_factor * r_facing_loss) + hit_factor * (
                       a - b), \
               b - a

    def apply(self, l_action, r_action):
        mt = 1
        mv = 1
        if len(l_action) != len(r_action):
            raise RuntimeError()
        for i in range(1):
            for i, k, j in zip(self.l_index, self.r_index,
                               range(len(l_action))):
                if p.getJointInfo(self.l, i, physicsClientId=self.client)[
                    2] == p.JOINT_PRISMATIC:
                    # print(float(p.getJointState(self.l, i,physicsClientId=self.client)[
                    #                           0] + (l_action[j] - 0.5) * mv), p.getJointState(self.l, i,physicsClientId=self.client)[
                    #                          0])
                    p.setJointMotorControl2(self.l, i, p.POSITION_CONTROL,
                                            p.getJointState(self.l, i,
                                                            physicsClientId=self.client)[
                                                0] + (l_action[j] - 0.5) * mv,
                                            force=100,
                                            physicsClientId=self.client)
                else:
                    # print(p.getJointState(self.l, i, physicsClientId=self.client)[0])
                    p.setJointMotorControl2(self.l, i, p.POSITION_CONTROL,
                                            p.getJointState(self.l, i,
                                                            physicsClientId=self.client)[
                                                0] + (l_action[j] - 0.5) * mt,
                                            force=100,
                                            physicsClientId=self.client)
                if p.getJointInfo(self.r, k, physicsClientId=self.client)[
                    2] == p.JOINT_PRISMATIC:
                    p.setJointMotorControl2(self.r, k, p.POSITION_CONTROL,
                                            p.getJointState(self.r, k,
                                                            physicsClientId=self.client)[
                                                0] + (r_action[j] - 0.5) * mv,
                                            force=100,
                                            physicsClientId=self.client)



                else:
                    p.setJointMotorControl2(self.r, k, p.POSITION_CONTROL,
                                            p.getJointState(self.r, k,
                                                            physicsClientId=self.client)[
                                                0] + (r_action[j] - 0.5) * mt,
                                            force=100,
                                            physicsClientId=self.client)
            for i in range(10):
                p.stepSimulation()
                a, b = self.get_contect()
                if a + b != 0:
                    break
                if self.gui:
                    time.sleep(0.01)
            if self.gui:
                print(l_action)
                print(r_action)

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
        left_prob, left_action = self.agent.apply_laction(state)
        right_state = self.get_right_state()
        right_prob, right_action = self.agent.apply_raction(right_state)
        self.apply(left_action, right_action)
        next_state = self.get_left_state()
        next_right_state = self.get_right_state()
        score, right_score, winner = self.get_score()
        mask = 1
        if winner != 0:
            mask = 0
            self.reset()
        return state, right_state, left_prob, left_action, right_prob, \
               right_action, next_state, next_right_state, score, right_score, winner, mask

    def step2(self,left_action,right_action):
        self.apply(left_action, right_action)
        score, right_score, winner = self.get_score()
        if winner != 0:
            self.reset()
        return score, winner


def get_data(g:Game,l:Actor_critic,r:Actor_critic,eproch=500,epslon=0.1):
    lstate = g.get_left_joint_pos()
    rstate = g.get_right_joint_pos()
    log_probs,scores,est_scores = [],[],[]
    for i in range(eproch):
        if(i%(eproch/100)==0):
            print(i/eproch*100,'%')
        la,lb,est_score = l(lstate,rstate)
        lvalue = (torch.rand(la.shape))
        if numpy.random.rand() > epslon:
            lvalue = torch.distributions.Beta(la + 0.000001,
                                         lb + 0.0000001).sample().detach()

        ra, rb, _ = r(rstate,lstate)
        rvalue = (torch.rand(la.shape))

        log_prob = torch.sum((la - 1) * torch.log(lvalue + 0.000001) + \
                (la - 1) * torch.log(1.000001 - lvalue),dim=-1)

        score, winner = g.step2(lvalue,rvalue)
        lstate = g.get_left_joint_pos()
        rstate = g.get_right_joint_pos()
        log_probs.append(log_prob)
        scores.append(score)
        est_scores.append(est_score)
    #print(log_probs)
    return torch.tensor(log_probs),torch.tensor(scores),torch.tensor(est_scores)

def trainstep(g:Game,l:Actor_critic,r:Actor_critic,max_step=100,epslon=0.1,gamma=0.99):
    log_probs,scores,est_stores = get_data(g,l,r,max_step,epslon)
    nest_set_scores = torch.roll(scores, -1, 0)
    nest_set_scores[-1,...]=0
    td = scores+nest_set_scores*gamma-nest_set_scores
    actor_loss = -log_probs*td.detach()
    return td,actor_loss

def train2(conv_hidden,hidden,action_length,lr:float,eprochs,max_step=100,epslon=0.1,gamma=0.99,td_factor=1):
    g = Game(None)
    l = Actor_critic(torch.tensor(adj_matrix),conv_hidden,hidden,action_length)
    all_saved=[]
    optimizer = torch.optim.Adam(l.parameters(),lr)
    l.train()
    for i in range(len(eprochs)):
        eproch= eprochs[i]
        all_saved.append(copy.deepcopy(l))
        r = random.choice(all_saved)
        print(i)
        for j in range(eproch):
            optimizer.zero_grad()
            td, actor_loss = trainstep(g,l,r,max_step,epslon,gamma)
            loss = td_factor* td+actor_loss
            print(td,actor_loss)
            loss.backward()
            optimizer.step()
        torch.save(l.state_dict(), 'model_dicts/actor.pth')

class Similater:
    def __init__(self, batch_size: int, agent: Agent,
                 drop_out_rate: float = 0.5):
        self.similations = []
        for i in range(batch_size):
            self.similations.append(Game(agent, drop_out_rate=drop_out_rate))

    def step(self, num_step=1, one_game=False):
        l_states = []
        r_states = []
        l_action = []
        r_action = []
        l_next_states = []
        r_next_states = []
        l_reward = []
        r_reward = []
        l_log_prob = []
        r_log_prob = []
        winners = []
        msks = []
        for samilator in self.similations:  # Step forward in time
            for _ in range(num_step):
                state, right_state, left_prob, left_action, right_prob, \
                right_action, next_state, next_right_state, score, right_score, winner, msk = samilator.step()
                l_states.append(list(state))
                r_states.append(list(right_state))
                l_action.append(left_action)
                r_action.append(right_action)
                l_next_states.append(list(next_state))
                r_next_states.append(list(next_right_state))
                l_reward.append(score)
                r_reward.append(right_score)
                l_log_prob.append(left_prob)
                r_log_prob.append(right_prob)
                winners.append(winner)
                msks.append(msk)
                if one_game and msk == 0:
                    break

            # p.stepSimulation()  # Real time simulation
        return torch.tensor(l_states), torch.stack(l_action), torch.tensor(
            l_next_states), torch.tensor(l_reward), torch.concatenate(
            l_log_prob), \
               torch.tensor(r_states), torch.stack(r_action), torch.tensor(
            r_next_states), torch.tensor(r_reward), torch.concatenate(
            r_log_prob) \
            , winners, msks


def sample_batch(simlater: Game, batch_size: int,
                 max_per_run: int = 1000, gamma: float = 0.99):
    rstates, raction, rnext_states, rreward, rlog_prob = [], [], [], [], []
    reward_to_go = []
    for i in range(batch_size):
        # print(i)
        data, states, action, next_states, reward, log_prob = simlater.get_answer(
            max_per_run)
        # print('d', torch.tensor(states) - torch.tensor(next_states))
        rstates, raction, rnext_states, rreward, rlog_prob = \
            rstates + states, raction + action, rnext_states + next_states, \
            rreward + reward, rlog_prob + log_prob
        reward_to_go = reward_to_go + compute_reward_to_go(reward, gamma)
    # print('rlp',torch.tensor(rlog_prob))
    return torch.tensor(rstates, dtype=torch.float32), torch.concatenate(
        raction), \
           torch.tensor(rnext_states, dtype=torch.float32), torch.tensor(
        rreward), \
           reward_to_go, torch.concatenate(rlog_prob)


def get_state(pid):
    # for i in range(p.getNumJoints(pid)):
    #   print(p.getJointInfo(pid,i))
    return p.getLinkState(pid, sward_top)[0] + p.getLinkState(pid, body)[0]


def get_loss(critic, states, next_states, reward, log_porb,
             normizalation_factor, gamma, msks):
    est_score = critic(states)
    est_next_score = critic(next_states)
    # print(reward.shape,est_next_score.shape,est_score.shape)
    td = reward + est_next_score.T * gamma * msks - est_score.T

    loss = torch.mean(td ** 2 + normizalation_factor * (
            est_next_score.T ** 2 + est_next_score.T ** 2))
    # reward_to_go = compute_reward_to_go(reward,gamma)
    # loss=torch.sum((torch.tensor(reward_to_go)- est_score.T)**2)
    actor_loss = -torch.sum(
        log_porb * td.detach())
    return loss, actor_loss


def train(actor: torch.nn.Module, r_actor: torch.nn.Module,
          critic: torch.nn.Module, gamma: float,
          s: Similater, epoch: int,
          critic_updates: int, actor_lr: int = 0.01, critic_lr: float = 0.001,
          normizalation_factor: float = 0.0001):
    critic_optimizer = torch.optim.Adam(lr=critic_lr,
                                        params=critic.parameters())
    actor_optimizer = torch.optim.Adam(lr=actor_lr, params=actor.parameters())
    r_actor_optimizer = torch.optim.Adam(lr=actor_lr,
                                         params=r_actor.parameters())

    critic.train()
    actor.train()
    r_actor.train()
    l = []
    a = []
    w = []
    for i in range(epoch):
        actor_optimizer.zero_grad()
        l_states, l_action, l_next_states, l_reward, l_log_porb, \
        r_states, r_action, r_next_states, r_reward, r_log_porb, \
        winners, msks = s.step(
            100, one_game=True)
        # msk = (reward == 0)
        msks = torch.tensor(msks)
        if i % (epoch // 100 + 1) == 0:
            print(int(i * 100 / epoch), '%')
        # td = 0
        # for _ in range(critic_updates):
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        r_actor_optimizer.zero_grad()

        l_critic_loss, l_actor_loss = get_loss(critic, l_states, l_next_states,
                                               l_reward, l_log_porb,
                                               normizalation_factor, gamma,
                                               msks)
        l_critic_loss.backward()
        critic_optimizer.step()
        l_actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        # r_critic_loss, r_actor_loss = get_loss(critic, r_states, r_next_states,
        #                                      r_reward, r_log_porb,
        #                                     normizalation_factor, gamma,
        #                                    msks)
        # r_critic_loss.backward()
        # critic_optimizer.step()
        # r_actor_loss.backward()
        # r_actor_optimizer.step()
        a.append(float(l_actor_loss))
        l.append(float(l_critic_loss))
        w.append(sum(winners) / len(winners))
        # torch.save(actor.state_dict(), f'model_dicts/actor{i}.pth')
        # torch.save(critic.state_dict(), f'model_dicts/critic{i}.pth')
    return l, a, w


'''
def train_actor(actor: torch.nn.Module, critic: torch.nn.Module, gamma: float,
                s: Similater, epoch: int,
                critic_updates: int, actor_lr: int = 0.01,
                critic_lr: float = 0.005,
                normizalation_factor: float = 0.0001):
    critic_optimizer = torch.optim.Adam(lr=critic_lr,
                                        params=critic.parameters())
    actor_optimizer = torch.optim.Adam(lr=actor_lr, params=actor.parameters())
    critic.train()
    actor.train()
    l = []
    a = []
    w = []
    states, action, next_states, reward, log_porb, winners, msks = s.step(100,one_game=True)
    for i in range(epoch):
        # msk = (reward == 0)
        actor_optimizer.zero_grad()
        msks = torch.tensor(msks)
        if i % (epoch // 100 + 1) == 0:
            print(int(i * 100 / epoch), '%')
        critic_optimizer.zero_grad()
        est_score = critic(states)
        est_next_score = critic(next_states)
        td = reward + est_next_score.T * gamma * msks - est_score.T
        l.append(torch.sum(est_score).detach())
        # print(est_score.T - reward_to_go)
        # print(log_porb.size())
        # print(td.size())
        # print(action.size())

        logProb = get_prob(actor, states, action)
        # print(logProb.shape)
        actor_loss = -torch.sum(logProb * (est_next_score - est_score).detach())
        actor_loss.backward()
        actor_optimizer.step()
        a.append(float(actor_loss))
        w.append(float(torch.sum(logProb * (est_next_score - est_score))))
        # torch.save(actor.state_dict(), f'model_dicts/actor{i}.pth')
        # torch.save(critic.state_dict(), f'model_dicts/critic{i}.pth')
    return l, a, w


def test_actor(state_length, action_length, critic_hidden, actor_hidden,
               actor_wights=None, critic_wights='model_dicts/critic3.pth'):
    critic = Critic(state_length,
                    critic_hidden)  # Critic(2 * s.l_last_state.size(0))
    gamma = 0.99
    actor = Actor(state_length, actor_hidden, action_length)
    if actor_wights != None:
        actor.load_state_dict(torch.load(actor_wights))
    if critic_wights != None:
        critic.load_state_dict(torch.load(critic_wights))
    agent = Agent(actor)

    s = Similater(1, agent, 1)

    epoch = 4
    critic_updates = 1
    normizalation_factor = 0.000001
    # print(action)
    l, a, w = train_actor(actor, critic, gamma, s, epoch, critic_updates,
                          normizalation_factor=normizalation_factor)

    # torch.save(actor.state_dict(), 'model_dicts/actor.pth')
    # torch.save(critic.state_dict(), 'model_dicts/critic.pth')
    # for i in range(reward.size(0)):
    #   est.append(float(critic(states[i])))

    # draw_buffer.append(float(critic(next_states[i])-reward_to_go[i])**2)
    print(l)
    plt.plot(l)
    plt.show()
    print(a)
    plt.plot(a)
    plt.show()
    print(w)
    plt.plot(w)
    plt.show()
    # plt.plot([sum(w[0:i])/i for i in range(1,len(w))])
    # plt.show()
    for i in s.similations:
        p.disconnect(i.client)

'''


def train_and_save(state_length, action_length, critic_hidden, actor_hidden,
                   actor_wights=None, critic_wights=None):
    critic = Critic(state_length,
                    critic_hidden)  # Critic(2 * s.l_last_state.size(0))
    gamma = 0.99
    actor = Actor(state_length, actor_hidden, action_length)
    if actor_wights != None:
        actor.load_state_dict(torch.load(actor_wights))
    if critic_wights != None:
        actor.load_state_dict(torch.load(critic_wights))
    agent = Agent(actor)

    s = Similater(1, agent, 1)

    epoch = 4000
    critic_updates = 1
    normizalation_factor = 0.000001
    # print(action)
    l, a, w = train(actor, critic, gamma, s, epoch, critic_updates,
                    normizalation_factor=normizalation_factor)

    torch.save(actor.state_dict(), 'model_dicts/actor.pth')
    torch.save(critic.state_dict(), 'model_dicts/critic.pth')
    # for i in range(reward.size(0)):
    #   est.append(float(critic(states[i])))

    # draw_buffer.append(float(critic(next_states[i])-reward_to_go[i])**2)
    print(l)
    plt.plot(l)
    plt.show()
    print(a)
    plt.plot(a)
    plt.show()
    print(w)
    plt.plot(w)
    plt.show()
    plt.plot([sum(w[0:i]) / i for i in range(1, len(w))])
    plt.show()
    for i in s.similations:
        p.disconnect(i.client)


def train_and_test(actor, critic, epoch, name='', test_itor=100):
    gamma = 0.99
    ractor = copy.deepcopy(actor)
    agent = Agent(actor, ractor)

    s = Similater(1, agent, 1)

    critic_updates = 10
    normizalation_factor = 0  # .000001
    # print(action)
    l, a, w = train(actor, ractor, critic, gamma, s, epoch, critic_updates,
                    normizalation_factor=normizalation_factor)

    torch.save(actor.state_dict(), f'model_dicts/nactor{name}.pth')
    torch.save(critic.state_dict(), f'model_dicts/ncritic{name}.pth')
    for i in s.similations:
        p.disconnect(i.client)
    s = Game(agent, False, 1)
    win = []
    all_scores = []
    est_scores = []
    actor_scores = []
    scores = []
    for i in range(test_itor):
        # p.stepSimulation()
        for j in range(100):
            state, right_state, left_prob, left_action, right_prob, \
            right_action, next_state, next_right_state, score, right_score, winner, mask = s.step()

            # state, left_prob, left_action, right_prob, right_action, next_state, score, winner, _ = s.step()
            scores.append(score)
            est = float(critic(state).detach())
            est_scores.append(est)
            actor_scores.append(float(-left_prob.detach() * est))
            if winner != 0:
                win.append(winner)
                print(i, winner)
                all_scores = all_scores + compute_reward_to_go(scores, 0.99)
                scores = []
                break
    all_scores = all_scores + compute_reward_to_go(scores, 0.99)
    # print('l',len(est_scores),len(all_scores))

    # time.sleep(0.01)
    p.disconnect()
    win_rate = (sum(win) / len(win) / 2 + 0.5) if len(win) > 0 else 0
    print(win_rate)
    return l, a, w, win_rate, all_scores, est_scores, actor_scores


def trian_and_test_all(state_length, action_length, critic_hidden, actor_hidden,
                       eporches=(1000, 1000, 1000, 1000), test_itor=100):
    all_l = []
    all_a = []
    all_w = []
    all_win_rate = []
    all_all_scores = []
    all_est_score = []
    all_actor_score = []

    actor = Actor(state_length, actor_hidden, action_length)
    actor.load_state_dict(torch.load('model_dicts/nactor13.pth'))
    critic = Critic(state_length, critic_hidden)
    for i in range(len(eporches)):
        print('epoch', i)
        epoch = eporches[i]
        l, a, w, win_rate, all_scores, est_scores, actor_scores = \
            train_and_test(actor, critic, epoch, name=str(i),
                           test_itor=test_itor)
        print('l', len(all_scores), len(est_scores))
        plt.plot(l)
        plt.savefig(f'./figers/critic_loss{i}')
        plt.cla()
        plt.plot(a)
        plt.savefig(f'./figers/actor_loss{i}')
        plt.cla()

        plt.plot(w)
        plt.savefig(f'./figers/win{i}')
        plt.cla()
        plt.plot(all_scores)
        plt.plot(est_scores)
        plt.savefig(f'./figers/pridicted_vs_actural{i}')
        plt.cla()
        plt.plot(actor_scores)
        plt.savefig(f'./figers/actor_score_in_test{i}')
        plt.cla()

        all_l.append(l)
        all_a.append(a)
        all_w.append(w)
        all_win_rate.append(win_rate)
        all_all_scores.append(all_scores)
        all_est_score.append(est_scores)
        all_actor_score.append(actor_scores)
    print(all_win_rate)
    a, e = [], []
    er = []
    for i in range(len(eporches)):
        print(i, len(all_all_scores[i]), len(all_est_score[i]))
        plt.plot(all_all_scores[i])
        plt.plot(all_est_score[i])
        a = a + list(all_all_scores[i])
        e = e + list(all_est_score[i])
        plt.show()
        er.append(np.sum((np.asarray(all_all_scores[i]) - np.asarray(
            all_est_score[i])) ** 2))
    plt.plot(a)
    plt.plot(e)
    plt.show()
    plt.plot(er)
    plt.show()
    '''
    for i in range(len(eporches)):
        print(i)
        plt.plot(all_l[i])
        plt.show()
        plt.plot(all_a[i])
        plt.show()
        plt.plot(all_w[i])
        plt.show()
        plt.plot(all_all_scores[i])
        plt.plot(all_est_score[i])
        plt.show()
        plt.plot(all_actor_score[i])
        plt.show()
    
    plt.plot(all_win_rate)
    plt.show()

    critic_error = []
    critic_var = []
    rtg_var = []
    action_var = []
    for i in range(len(eporches)):
        rtg_var.append(np.var(all_all_scores[i]))
        critic_var.append(np.var(all_est_score[i]))
        print(numpy.asarray(all_all_scores[i]),
              numpy.asarray(all_all_scores[i]).size)
        print(numpy.asarray(all_est_score[i]),
              numpy.asarray(all_est_score[i]).size)
        # critic_error.append(np.sum((numpy.asarray(all_all_scores[i])-numpy.asarray(all_est_score[i]))**2))
        action_var.append(np.var(all_actor_score[i]))
    plt.plot(critic_var)
    plt.plot(rtg_var)
    plt.show()
    plt.plot(action_var)
    plt.show()
    # plt.plot(critic_error)
    # plt.show()
'''


def play(state_length, action_length, critic_hidden, actor_hidden, i1, i2):
    actor = Actor(state_length, actor_hidden, action_length)
    actor.load_state_dict(torch.load(f'model_dicts/nactor{i1}.pth'))
    ractor = Actor(state_length, actor_hidden, action_length)
    ractor.load_state_dict(torch.load(f'model_dicts/nactor{i2}.pth'))
    agent = Agent(actor, ractor, 0)
    s = Game(agent, True, 0)
    win_time = 0
    total_time = 0
    # while 1:
    while 1:
        '''
        p.stepSimulation()
        (s.get_score())
        '''
        state, right_state, left_prob, left_action, right_prob, \
        right_action, next_state, next_right_state, score, right_score, winner, mask = s.step()
        if winner != 0:
            win_time = win_time + (winner + 1) / 2
            total_time += 1
            print(win_time / total_time)
        print(score, right_score)

    p.disconnect()


def check(state_length, action_length, critic_hidden, actor_hidden):
    actor = Actor(state_length, actor_hidden, action_length)
    # actor.load_state_dict(torch.load('model_dicts/actor.pth'))
    agent = Agent(actor)
    s = Game(agent, True, 0)
    while 1:
        p.stepSimulation()
        # print(get_curr_state(s.l,s.client))
        # print(s.get_score())
        time.sleep(0.01)
    p.disconnect()


def test(state_length, action_length, critic_hidden, actor_hidden):
    actor = Actor(state_length, actor_hidden, action_length)
    actor.load_state_dict(torch.load('model_dicts/actor9.pth'))
    agent = Agent(actor)
    critic = Critic(state_length, critic_hidden)
    critic.load_state_dict(torch.load('model_dicts/critic9.pth'))
    s = Game(agent, False, 1)
    win = []
    all_scores = []
    est_scores = []
    actor_scores = []
    for i in range(1):
        # p.stepSimulation()
        scores = []
        while 1:
            state, left_prob, left_action, right_prob, right_action, next_state, score, winner, _ = s.step()
            scores.append(score)
            est = float(critic(state).detach())
            est_scores.append(est)
            actor_scores.append(float(-left_prob.detach() * est))
            if winner != 0:
                win.append(winner)
                print(i, winner)
                all_scores = all_scores + compute_reward_to_go(scores, 0.99)
                break

        # time.sleep(0.01)
    p.disconnect()
    print(sum(win) / len(win) / 2 + 0.5)
    plt.plot(all_scores)
    plt.plot(est_scores)
    plt.show()
    plt.plot(actor_scores)
    plt.show()


def model_selection(state_length, action_length, critic_hidden, actor_hidden):
    actor = Actor(state_length, actor_hidden, action_length)
    actor.load_state_dict(torch.load('model_dicts/actor.pth'))
    agent = Agent(actor)
    critic = Critic(state_length, critic_hidden)
    critic.load_state_dict(torch.load('model_dicts/critic.pth'))
    s = Game(agent, False, 1)
    est_scores = []
    all_scores = []
    states = []
    next_states = []
    for _ in range(50):
        scores = []
        while 1:
            state, left_prob, left_action, right_prob, right_action, next_state, score, winner, _ = s.step()
            states.append(list(state))
            next_states.append(list(next_state))
            scores.append(score)
            print(winner)
            if winner != 0:
                all_scores = all_scores + compute_reward_to_go(scores, 0.99)
                break
    p.disconnect()
    ests = []
    plt.plot(all_scores, 'b')
    for i in range(1000):
        critic.load_state_dict(torch.load(f'model_dicts/critic{i}.pth'))
        all_scores = torch.tensor(all_scores)
        est = critic(torch.tensor(states)).detach().T[0]
        print(est)
        ests.append(torch.sum(
            (critic(torch.tensor(states)).detach() - all_scores) ** 2))
        if i % 10 == 0:
            plt.plot(est, 'r')
    plt.show()
    plt.plot(ests)
    plt.show()


if __name__ == '__main__':

    state_length = len(state_points) * 6 - 3
    action_length = 17
    critic_hidden = 3000
    actor_hidden = 900
    # a,b,c,d =[-0.65082332,  1.70149551,  0.49605632], [-0.15045156,  1.63369178, -0.80971195], [-1.47627926,  1.44627016,  0        ], [-1.47627926,  1.44627016,  1.5       ]
    # print(shortest_distence(np.asarray(a), np.asarray(b),np.asarray(c), np.asarray(d)))
    # model_selection(state_length, action_length, critic_hidden,actor_hidden)
    # train_and_save(state_length, action_length, critic_hidden,actor_hidden)
    # test(state_length, action_length, critic_hidden,actor_hidden)
    # test_actor(state_length, action_length, critic_hidden,actor_hidden)
    #play(state_length, action_length, critic_hidden, actor_hidden, 23, 23)
    # check(state_length, action_length, critic_hidden,actor_hidden)

    #trian_and_test_all(state_length, action_length, critic_hidden, actor_hidden,
     #                  eporches=(100,) * 40, test_itor=10)
    conv_hidden, hidden = 5,900
    train2(conv_hidden, hidden, action_length, 0.01,(100,)*5)

    # check(state_length, action_length, hidden)

'''
    client = p.connect(p.GUI)
    l = p.loadURDF('models/two_arms.urdf', physicsClientId=client,flags=p.URDF_USE_SELF_COLLISION)

    p.resetJointState(l, 0, 0, physicsClientId=client)
    p.resetJointState(l, 1, 0, physicsClientId=client)
    p.resetJointState(l, 2, 0, physicsClientId=client)
    p.resetJointState(l, 4, 3.14 / 4, physicsClientId=client)
    p.resetJointState(l, 13, 3.14 / 2, physicsClientId=client)

    #r = p.loadURDF('models/two_arms.urdf', physicsClientId=client,
     #                   flags=p.URDF_USE_SELF_COLLISION)
    #p.resetJointState(r, 0, 3, physicsClientId=client)
    #p.resetJointState(r, 1, 0, physicsClientId=client)
    #p.resetJointState(r, 2, 3.14, physicsClientId=client)
    #p.resetJointState(r, 4, 3.14 / 4, physicsClientId=client)
    #p.resetJointState(r, 13, 3.14 / 2, physicsClientId=client)
    i=0
    print(p.getJointInfo(l, i))
    print(p.getLinkState(l, i))
    print(p.getJointState(l, i))
    print(p.getJointInfo(l, i)[13])
    p.changeVisualShape(l, i, rgbaColor=[0, 0, 1])
    for _ in range(500):
        p.applyExternalForce(l, i, np.asarray(p.getJointInfo(l, i)[13]) * 300, [0,0,0],
                              p.LINK_FRAME)
        print(p.getJointState(l,i))
        # p.applyExternalTorque(r,5,(-50,0,0),p.WORLD_FRAME)
        p.stepSimulation()
        time.sleep(0.01)
    for _ in range(500):
        p.applyExternalTorque(l, i, np.asarray(p.getJointInfo(l, i)[13]) * 10,
                              p.LINK_FRAME)
        # print(p.getJointState(l,i))
        # p.applyExternalTorque(r,5,(-50,0,0),p.WORLD_FRAME)
        p.stepSimulation()
        time.sleep(0.01)
    p.changeVisualShape(l, i, rgbaColor=[1, 0, 0])
'''
