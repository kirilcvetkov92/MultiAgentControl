# Project 2L Continuous Control

### Project details

For this project, we show how to train an agent(double-jointed arm) to move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.  

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.  
Youtube video : 

<p align="center">
    <a href = "https://www.youtube.com/watch?v=4sId1_9EkR0"> <img src="https://img.youtube.com/vi/4sId1_9EkR0/0.jpg" /></a>
</p>



### Instructions
* To train the agent run ```python Navigation.py```
* To run the trained agent run ```python play.py```

#### Saved Model Weights
The submission includes the saved model weights of the successful agent.  
Actor checkout : ```checkpoint_actor.pth```
Critic checkout : ```checkpoint_critic.pth```

### Report
To train the agent run python Navigation.py

## Implementation details

This problem was solved with DDPQ Neural network 

I choose the '20 agents' environment because we can collect more data in order to provide more accurate and better-quality result.

DDPQ (deep deterministic plicy gradient) reinforcement learning approach was chosen to sole this task.

The DDPQ network model consists od 2 parts:

* Actor Network
* Critic Network

## Actor Network

* This network takes the state as an input and output the policy π | π = S->A 
* The 'gain' estimated function is actually the Q value, and that function is approcimated by the Critic network.
  The 'goal' of this network is to find the correspodning action, when a state is given.
  this neural network favorize the action which maximize the Q value as otuput by the critic network.
  Since this is a problem maximization of the Q-value, the optimization problem will be gradiend ascent.
  Since the action is continious, the input state vector is 33 dimensional, I needed to create neural networks with bigger 
  hidden size inside the layers.
  A 2-layer fully connected dense with 256 neurons on the first layer and 128 neurons on second layer was chosen.
  The actions were clipped in range of (-1, 1) and tanh activation fucntion was used in the last layer.
  
  
## Critic Network

The critic neural network in this problem serves as a 'judge' to balance the estimation made by the actor network.
While the Actor network has obligation to maximize the Q value, the critic network is trying to 'make sure' that 
the provided estimation by the Actor is accurate.
The way that the critic network is performing is the following : 
It takes the mean square loss of the td error (TDTarget - TDCurrent).
This ensures that the DDPG will not be subject too much variance.
I choose very similar architecture with the actor neural network, so I used 256 neurons for the first layer and 128 neurons for the second layer.
This neural network receive the (state, action) pair as an input and it outputs the estimated Q value.

## Training process
The training depends on balancing between adctor and critic network.
It's very important to choose a similar architecture and hyperparameters between these 2 neural networks.
If one network is super powerful compared to the other, then the training would fail..
Here I also found that using ELU insteaed of RELU layers provides more stable training.
Gradient clipping was applied for the critic neural network, to provide more stable performance.



Here is a summary of the hyper parameters used:

<table width=600>
<tr><td>Memory buffer size  </td><td> 1e6    </td></tr>   
<tr><td>L2 Weight Decay  </td><td>  0    </td></tr>
<tr><td>Gamma  </td><td> 0.99    </td></tr>               
<tr><td>Tau (soft update)  </td><td> 1e-3          </td></tr>           
<tr><td>Learning Rate Actor </td><td>  1e-4  </td></tr>
<tr><td>Learning Rate Critic  </td><td>  1e-4  </td></tr>
<tr><td>Learning network frequency </td><td> 20    </td></tr>
<tr><td>Learning times per step  </td><td> 10    </td></tr>
<tr><td> Sigma </td><td> 0.1   </td></tr>

</table>



##### Plot of Rewards

<p align="center">
    <img src="documentation/Scores.png" />
</p>

* Environment solved in 105 episodes

