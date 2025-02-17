import kaggle_environments
import gym
import numpy as np
list =[[1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8]]
#if list[1,5]:
#    print(truwuw)

env = kaggle_environments.make("connectx", debug=True)
out = env.render(mode="ansi")
trainer = env.train([None,"negamax"])
board=trainer.reset()["board"]
print(board)
resh_board = np.array(board).reshape(6,7)
print(resh_board)