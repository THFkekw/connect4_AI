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

def getReward(winner,state,invalidAction,turns):
    if not state:
        reward = 0

    if winner == 1:
        reward = 1000 #* 25/turns
    elif winner == 0:
        reward = -1000
    else:
        reward = -10000

    #reward += turns * 3

    if not invalidAction:
        #reward = -50
        pass


    return reward

def train_step(model, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
      #Propagate through the agent network
        logits = model(observations)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
        loss = tf.reduce_mean(softmax_cross_entropy * rewards)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 

def checkValid(action_,board,only_check=False,only_return=False):
    valid = True
    action  = 0
    action = action_
    resh_board = np.array(board).reshape(6,7)
    
    if resh_board[0][action] != 0:
        valid = True
        for i in range (20):
            if action >6:
                    action = 0
            if resh_board[0][action] != 0:
                
                action +=1  
                valid = False     
            else:
                
                if only_check:
                    return valid
                elif only_return:
                    return action
                else:
                    return action,valid
    
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
    #model.add(layers.Dense(100,activation="relu"))
    model.add(layers.Dense(448,activation="relu"))
    model.add(layers.Dense(112,activation="relu"))
    model.add(layers.Dense(28,activation="relu"))
    #model.add(layers.Dense(5,activation="relu"))
    #model.add(layers.Dense(5,activation="relu"))
    #model.add(layers.Dense(5,activation="relu"))
    #model.add(layers.Dense(5,activation="relu"))
    #model.add(layers.Dense(5,activation="relu"))
    #model.add(layers.Dense(1000,activation="relu"))

    #model.add(layers.Dense(1000,activation="relu"))
    #model.add(layers.Dense(100,activation="relu"))
    model.add(layers.Dense(7,activation="linear"))
    return model

def saveModel(model):
    pass

def loadModel():
    pass

def testModel(model,rounds):
    pass

def train_model(model,rounds,verbose=False,adversary="random"):
    #Train P1 (model) against random agent P2
    #Create the environment
    env = make("connectx", debug=True)
    #Optimizer
    optimizer = tf.keras.optimizers.Adam()
    #Set up the experience storage
    exp = Experience()
    epsilon = 1
    epsilon_rate = 0.99
    wins = 0
    win_track = []
    i = 0
    epsilon_cap = 0.005
    turns = 0
    
    for episode in range(rounds):
        print(i)
        i +=1
        #Set up random trainer
        trainer = env.train([None, adversary])
        #First observation
        obs = np.array(trainer.reset()['board']).reshape(6,7)
        #Clear cache
        exp.clear()
        #Decrease epsilon over time if we want
        epsilon = epsilon * epsilon_rate
        if epsilon < epsilon_cap:
            epsilon = epsilon_cap
        #Set initial state
        state = False
        turns = 0
        while not state:

            #Get action
            action, w = getAction(model, obs, epsilon)
            #Check if action is valid
            while True:
                temp_action = action
                #print(temp_action)
                action,valid = checkValid(temp_action,obs)
                if temp_action == action:
                    break
            #Play the action and retrieve info
            new_obs, winner, state, info = trainer.step(action)
            #print(winner)
            obs = np.array(new_obs['board']).reshape(6,7)
            turns += 1
            #Get reward
            reward = getReward(winner, state,valid,turns)
            #Store experience
            exp.store_experience(obs, action, reward)
            #Break if game is over
            if state:
                #This would be where training step goes I think
                if winner == 1:
                    wins += 1
                win_track.append(winner)
                train_step(model, optimizer = optimizer,
                                observations = np.array(exp.observations),
                                actions = np.array(exp.actions),
                                rewards = exp.rewards)
                if verbose:
                    print(env.render(mode="ansi"))
                break
            if verbose:
                print(env.render(mode="ansi"))
    print(win_track)

def main(rounds):
    model = make_model()
    train_model(model,rounds)
    train_model(model,rounds,adversary="negamax")

if __name__ =="__main__":
    rounds =  500
    main(rounds)
