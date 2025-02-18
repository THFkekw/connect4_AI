import numpy as np
import tensorflow as tf
from kaggle_environments import evaluate, make, utils
import keras
from keras import layers
import random
import gym


#Define epsilon decision function
def epsilonDecision(epsilon):
  action_decision = random.choices(['model','random'], weights = [1 - epsilon, epsilon])[0]
  return action_decision
epsilonDecision(epsilon = 0) # would always give 'model'

def getAction(model, observation, epsilon):
    #Get the action based on greedy epsilon policy
    action_decision = epsilonDecision(epsilon)
    #Reshape the observation to fit in model
    observation = np.array([observation])
    #Get predictions
    preds = model.predict(observation)
    #Get the softmax activation of the logits
    weights = tf.nn.softmax(preds).numpy()[0]
    if action_decision == 'model':
        action = np.argmax(weights)
    if action_decision == 'random':
        action = random.randint(0,6)
    return int(action), weights

def makeMove(observation):
    observation = np.array([observation])
    preds = model.predict()
    weights = tf.nn.softmax(preds).numpy()[0]
    action = np.argmax(weights)
    return action

def getReward(winner,state,invalidAction):
    if not state:
        reward = 0

    if winner == 1:
        reward = 100
    elif winner == 0:
        reward = 0
    else:
        reward = -100

    if not invalidAction:
        reward = -5

    return reward

def train_step(model, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
      #Propagate through the agent network
        logits = model(observations)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
        loss = tf.reduce_mean(softmax_cross_entropy * rewards)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 

def checkValid(action,board,only_check=False,only_return=False):
    valid = True
    resh_board = np.array(board).reshape(6,7)
    if resh_board[0][action] != 0:
        action = random.randint(0,6)
        if resh_board[0][action] == action:
            if action == 0:
                action += 1
            elif action == 6:
                action -=1
            else:
                action +=1
        valid = False
    if only_check:
        return valid
    elif only_return:
        return action
    else:
        return action,valid

class Experience:
    def __init__(self):
        self.clear() 
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
    def store_experience(self, new_obs, new_act, new_reward):
        self.observations.append(new_obs)
        self.actions.append(new_act)
        self.rewards.append(new_reward)

def make_model():
    model = keras.Sequential()
    model.add(layers.Input(shape=[6,7]))
    model.add(layers.Flatten())
    model.add(layers.Dense(50,activation="relu"))
    model.add(layers.Dense(50,activation="relu"))
    model.add(layers.Dense(50,activation="relu"))
    model.add(layers.Dense(50,activation="relu"))
    model.add(layers.Dense(7,activation="linear"))
    return model

def train_model(model):
    #Train P1 (model) against random agent P2
    #Create the environment
    env = make("connectx", debug=True)
    #Optimizer
    optimizer = tf.keras.optimizers.Adam()
    #Set up the experience storage
    exp = Experience()
    epsilon = 0.2
    epsilon_rate = 1
    wins = 0
    win_track = []
    i = 0
    for episode in range(100):
        print(i)
        i +=1
        #Set up random trainer
        trainer = env.train([None, 'negamax'])
        #First observation
        obs = np.array(trainer.reset()['board']).reshape(6,7)
        #Clear cache
        exp.clear()
        #Decrease epsilon over time if we want
        epsilon = epsilon * epsilon_rate
        #Set initial state
        state = False
        while not state:
            #Get action
            action, w = getAction(model, obs, epsilon)
            #Check if action is valid
            while True:
                temp_action = action
                action,valid = checkValid(temp_action,obs)
                if temp_action == action:
                    break
            #Play the action and retrieve info
            new_obs, winner, state, info = trainer.step(action)
            obs = np.array(new_obs['board']).reshape(6,7)
            #Get reward
            reward = getReward(winner, state,valid)
            #Store experience
            exp.store_experience(obs, action, reward)
            #Break if game is over
            if state:
                #This would be where training step goes I think
                if winner == 1:
                    wins += 1
                win_track.append(wins)
                train_step(model, optimizer = optimizer,
                                observations = np.array(exp.observations),
                                actions = np.array(exp.actions),
                                rewards = exp.rewards)
                print(env.render(mode="ansi"))
                break
            print(env.render(mode="ansi"))
    print(win_track)

def main():
    train_model(make_model())

if __name__ =="__main__":
    tf.test.is_built_with_cuda()
    tf.test.is_gpu_available(cuda_only=True)
    main()
