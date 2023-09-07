import time

import torch
from matplotlib import pyplot as plt
import pybullet as p
from game import get_joint_limits

class AutoEncoder(torch.nn.Module):
    def __init__(self, input, hidden):
        super().__init__()
        self. l1 = torch.nn.Linear(input,hidden)
        self.l2 = torch.nn.Linear(hidden, input)
        self.activation = torch.nn.ReLU()

    def forward(self,x):
        return self.l2(self.activation(self.l1(x)))


def train(input, hidden_size,eporch):
    loss = []
    val_loss=[]
    split = int(input.size(0)*0.7)
    training = input[0:split]
    valdation = input[split:]
    encoder = AutoEncoder(input.size(1),hidden_size)
    optimizer = torch.optim.Adam(encoder.parameters(),lr = 0.01)
    for i in range(eporch):
        encoder.train()
        output = encoder(training)
        l = torch.sum((training-output)**2)
        loss.append(l)
        l.backward()
        optimizer.step()
        val_loss.append((valdation-encoder(valdation))**2)
    plt.plot(val_loss)
    plt.plot(loss)
    plt.show()

def get_data(batch_size:int):
    client = p.connect(p.DIRECT)
    l = p.loadURDF('models/two_arms.urdf', physicsClientId=client,
                        flags=p.URDF_USE_SELF_COLLISION)

    lower, upper, index = get_joint_limits(l,client)
    diff = upper-lower
    data = []
    for i in range(batch_size):
        for i, j in zip(index, range(len(index))):
            pass

if __name__=='__main__':
    p.connect(p.GUI)
    l = p.loadURDF('models/test.urdf')
    while 1:
        p.applyExternalTorque(l,-1,(0,1,0),p.WORLD_FRAME)
        p.stepSimulation()
        time.sleep(0.01)




