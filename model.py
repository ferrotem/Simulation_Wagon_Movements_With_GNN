import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import gym
env_name = "CartPole-v1"
env = gym.make(env_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

############# Preprecossing Part #############

image = torch.rand((210,160,128))
# im_trans = transforms.functional.to_grayscale(image)

pre_process = transforms.Compose([ 
        transforms.ToPILImage(),
         transforms.Grayscale(num_output_channels=1), 
         transforms.Resize((110,84)),
         transforms.CenterCrop(84),
         transforms.ToTensor()
        ])

# we need to get last 4 frames
# [84,84,4]


#The outputs correspond to the predicted Q-values of the individual
#action for the input state.

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8,8), stride=4),
            nn.ReLU(),
            nn.Conv2d(16, out_channels=32, kernel_size=(4,4), stride=2),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=256),
            nn.ReLU(),
            nn.Linear(256,2))

    def forward(self,x):
        x = self.ConvNet(x)
        return x

net = DQN()

def function_phi(list_img):
    if len(list_img)<4:
        list_img = [list_img[0],list_img[0],list_img[0],list_img[0]]
    input_tensor = torch.cat([pre_process(x) for x in list_img])
    input_tensor =input_tensor.permute(1,2,0)
    return input_tensor


eps = 0.01
random_action = torch.bernoulli(eps)
action_list = env.action_space.n
N_episode = 10
sequence = []
for episode in range(N_episode):
    state = env.reset()
    image = env.render(mode="rgb_array")
    phi = function_phi(image)
    sequence = image
    while not done:
        random_action = torch.bernoulli(eps)
        if random_action==1:
           action  = [torch.randint(len(action_list), (10,))] 
        else:
            
            action = torch.argmax(net(phi))
        
        # Here we execute action in emulator and get reward r_t and image_t_1
        state, reward, done, info = env.step(action)

        image = env.render(mode="rgb_array")
        sequence = [sequence, action,image]
        phi = function_phi()

