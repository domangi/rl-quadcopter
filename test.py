## TODO: Train your agent here.
import numpy as np
import sys
from agents.ddpg.ddpg import DDPG
from my_task import MyTask

num_episodes = 1000
init_pos = np.array([0., 0., 0., 0., 0., 0.])
target_pos = np.array([0., 0., 100.])
task = MyTask(init_pose=init_pos, target_pos=target_pos,runtime=5.)
agent = DDPG(task)
rewards = []

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    step = 0
    while True:
        step +=1
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if step%2==0:
            print("\r\nEp={:4d}, score={:7.3f} (best={:7.3f}) pos={} {} {}, dist={}".format(
                i_episode,
                agent.score,
                agent.best_score,
                round(task.sim.pose[:3][0],2),
                round(task.sim.pose[:3][1],2),
                round(task.sim.pose[:3][2],2),
                abs(task.sim.pose[:3] - task.target_pos).sum()), end="")  # [debug]
        if done:
            rewards.append(agent.score)
            print("\r\n\n************")
            print("\r\nEp={:4d}, score={:7.3f} (best={:7.3f}) pos={} {} {}, dist={}".format(
                i_episode,
                agent.score,
                agent.best_score,
                round(task.sim.pose[:3][0],2),
                round(task.sim.pose[:3][1],2),
                round(task.sim.pose[:3][2],2),
                abs(task.sim.pose[:3] - task.target_pos).sum()), end="")  # [debug]
            print("\r\n\n************")
            break
    sys.stdout.flush()
