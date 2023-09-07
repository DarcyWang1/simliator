import copy

import numpy.random
import torch


class Actor(torch.nn.Module):
    def __init__(self, state_length, hidden, action_length):
        super().__init__()
        self.a = torch.nn.Linear(hidden, action_length)
        self.b = torch.nn.Linear(hidden, action_length)
        self.l1 = torch.nn.Linear(state_length,hidden)
        self.l2 = torch.nn.Linear(hidden, 2*hidden)
        self.l3 = torch.nn.Linear(2*hidden,hidden)
        self.actavation_1 = torch.nn.Sigmoid()
        self.actavation = torch.nn.ReLU()


    def forward(self, x):
        x = self.actavation(self.l1(x))
        x = self.actavation(self.l2(x))
        x = self.actavation(self.l3(x))
        return self.actavation_1(self.a(x)), self.actavation_1(
            self.b(x))


class Critic(torch.nn.Module):
    def __init__(self, state_length, hidden):
        super().__init__()
        self.layers =torch.nn.Sequential(
        torch.nn.Linear(state_length, hidden, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 2 * hidden, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(2*hidden, 1))

    def forward(self, state):
        # print(state.type)
        return self.layers(state)


class grach_conv(torch.nn.Module):
    def __init__(self, adj_matrix:torch.Tensor, conv_hidden):
        super().__init__()
        n = len(adj_matrix)
        self.adj_matrix = adj_matrix.float()
        self.layers = []
        conv_hidden=[3]+conv_hidden
        print(conv_hidden)
        for i in range(1,len(conv_hidden)):
            self.layers.append(torch.nn.Linear(conv_hidden[i-1], conv_hidden[i]))
        self.actavation = torch.nn.ReLU()
    def forward(self,x):
        #x = torch.moveaxis(x, -2, -1).float()
        for i in self.layers:
            x = self.adj_matrix@x#x @ self.adj_matrix.T

            x=i(x)
            x=self.actavation(x)
        return torch.moveaxis(x, -2, -1)

class Actor_critic(torch.nn.Module):
    def __init__(self, adj_matrix, conv_hidden, hidden, action_length):
        super().__init__()
        n = len(adj_matrix)
        self.adj_matrix=adj_matrix
        self.lc = grach_conv(self.adj_matrix,[conv_hidden,2*conv_hidden,4*conv_hidden])
        self.rc = grach_conv(self.adj_matrix,[conv_hidden,2*conv_hidden,4*conv_hidden])

        state_length = 2*4*conv_hidden*n
        self.a = torch.nn.Linear(hidden, action_length)
        self.b = torch.nn.Linear(hidden, action_length)
        self.critic_out = torch.nn.Linear(hidden,1)
        self.l1 = torch.nn.Linear(state_length,hidden)
        self.l2 = torch.nn.Linear(hidden, 2*hidden)
        self.l3 = torch.nn.Linear(2*hidden,hidden)
        self.actavation_1 = torch.nn.Sigmoid()
        self.actavation = torch.nn.ReLU()



    def forward(self, l,r):
        l =self.lc(l)
        r = self.rc(r)
        #print('a',l.shape,r.shape)
        x= torch.concatenate([torch.flatten(l,-2,-1),torch.flatten(r,-2,-1)],dim=-1)
        #print(x.shape)
        x = self.actavation(self.l1(x))
        x = self.actavation(self.l2(x))
        x = self.actavation(self.l3(x))
        return self.actavation_1(self.a(x)), self.actavation_1(
            self.b(x)), self.critic_out(x)

class Agent:
    def __init__(self, lactor:Actor,ractor=None,epslon=0.3):
        self.lactor = lactor
        if ractor is not None:
            self.ractor=ractor
        else:
            self.ractor = copy.deepcopy(lactor)
        self.epslon=epslon
    def action_prob(self,self_action_a, self_action_b):
        '''
        value = (1 - torch.rand(
            self_action_a.size()[0]) ** self_action_b) ** self_action_a
        log_prob = torch.log(self_action_a + 0.00001) + torch.log(
            self_action_b + 0.00001) + \
                   (self_action_a - 1) * torch.log(value.detach() + 0.00001) + \
                   (self_action_b - 1) * torch.log(
            1.00001 - value.detach() ** self_action_a)
        '''
        if numpy.random.rand() >self.epslon:
            value = torch.distributions.Beta(self_action_a+0.000001,self_action_b+0.0000001).sample().detach()
        else:
            value = (torch.rand(self_action_a.shape))
            #print('v',value)
        log_prob = (self_action_a-1)*torch.log(value+0.000001)+\
                   (self_action_b-1)*torch.log(1.000001-value)
        # 1+1/a, b, since self_action_a = 1/a, self_action_b = 1/b
        return torch.sum(log_prob).resize(1), value
    def apply_laction(self, x):
        #print(x)
        self_action_a, self_action_b = self.lactor(x)
        # x = (1 - u^(1/b))^(1/a)
        return self.action_prob(self_action_a, self_action_b)

    def apply_raction(self, x):
        self_action_a, self_action_b = self.ractor(x)
        # x = (1 - u^(1/b))^(1/a)
        return self.action_prob(self_action_a, self_action_b)
    def get_actors(self):
        return self.lactor, self.ractor


    def get_prob(self,state,value):
        self_action_a, self_action_b = self.lactor(state)
        #print(value)
        log_prob = (self_action_a - 1) * torch.log(value + 0.000001) + \
                   (self_action_b - 1) * torch.log(1.000001 - value)
        return torch.sum(log_prob,dim=-1)

def get_action(self_action_a, self_action_b):
    return (1 - torch.rand(self_action_a.size()[0]) ** self_action_b) ** self_action_a
