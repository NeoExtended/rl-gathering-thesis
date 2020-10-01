"""
  gym id: Maze0122Env-v0
  Input: gray-scale images
  Reward: region range basis (easy)
  Render: gray-scale visualization
  Training: from scratch
"""
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np, random, sys, matplotlib.pyplot as plt, time, os
import gym
from gym import error, spaces, utils, core
from gym.utils import seeding

from time import sleep
plt.ion()



class Maze0122Env(core.Env):
    def __init__(self):
        global mazeData, costData, freespace, mazeHeight, mazeWidth, robot_marker
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.map_data_dir = dir_path+'/../../mapdata'

        robot_marker = 150
        self.goal_range = 20
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7] # {N, NE, E, SE, S, SW, W, NW}
        # self.action_dict = {0: (-1, 0), 1: (1, 0), 2: (-1, 1), 3: (1, 1)}  # {up, down, left ,right}
        self.action_map = {0: (1, 0),  1: (1, 1),  2: (0, 1), 3: (-1, 1),
                           4: (-1, 0), 5: (-1,-1), 6: (0,-1), 7: (1, -1)}
        self.rev_action_map = {(1, 0):0, (1, 1):1, (0, 1):2, (-1, 1):3,
                           (-1, 0):4, (-1, -1):5, (0, -1):6, (1, -1):7}
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)
        mazeData, costData, freespace = self._load_data(self.map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
        self.seed()
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.freespace = np.ones((mazeHeight, mazeWidth))-freespace

        self.goal = np.array([204, 96])
        self.init_state = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _load_data(self, data_directory):
        filename = 'map0122'
        mazeData = np.loadtxt(data_directory + '/'+filename+'.csv').astype(int)
        freespace = np.loadtxt(data_directory + '/'+filename+'_freespace.csv').astype(int)
        costData = np.loadtxt(data_directory + '/' +filename+ '_costmap.csv').astype(int)
        return mazeData, costData, freespace

    def _build_robot(self):
      # ======================
      # For transfer learning only
        self.tflearn = False
        self.cur_robot = None
      # ======================
        row, col = np.nonzero(freespace)
        self.reward_grad = np.zeros(40).astype(np.uint8)
        self.robot_num = 256 #len(row)
        self.robot = random.sample(range(row.shape[0]), self.robot_num)
        self.state = np.zeros(np.shape(mazeData)).astype(int)
        self.state_img = np.copy(self.state)
        self.loc = np.zeros([self.robot_num,2]).astype(np.int32)
        for i in range(self.robot_num):
            self.loc[i, :] = row[self.robot[i]], col[self.robot[i]]
            self.state[row[self.robot[i]], col[self.robot[i]]] += robot_marker
            self.state_img[row[self.robot[i]]-1:row[self.robot[i]]+2,
                col[self.robot[i]]-1:col[self.robot[i]]+2] = robot_marker*np.ones([3,3])

        self.init_state = self.state
        self.init_state_img = self.state_img

        self.output_img = self.state_img + self.maze * 255

        return (np.expand_dims(self.output_img, axis=2))

    def get_costmap(self):
        return (np.expand_dims(costData, axis=2))
        
    def step(self, action):

        info = {}

        # =====================
        # Transfer learning
        if self.tflearn:
          action = self.instructor()
          info = {'ac': action}
        # =====================

        dy, dx = self.action_map[action]

        prev_loc = np.copy(self.loc)
        self.loc = np.add(self.loc, np.array([dy, dx]))
        collision = np.where(self.freespace[self.loc[:, 0], self.loc[:, 1]] == 1.0)
        self.loc[collision, :] = prev_loc[collision, :]

        self.state_img  *= 0 # np.zeros([mazeHeight,mazeWidth])

        for i in range(self.robot_num):
            self.state_img[self.loc[i,0]-1:self.loc[i,0]+2, self.loc[i,1]-1:self.loc[i,1]+2] = robot_marker * np.ones([3, 3])

        self.output_img = self.state_img + self.maze*255

        done, reward = self.get_reward()

        return(np.expand_dims(self.output_img,axis=2),reward,done,info)

    def get_reward(self):
        cost_to_go = np.sum(costData[self.loc[:, 0], self.loc[:, 1]])

        max_cost_agent = np.max(costData[self.loc[:, 0], self.loc[:, 1]])

        done = False
        reward = 0.0

        if max_cost_agent <= self.goal_range:
            done = True
            reward += 500
            return done, reward
        elif max_cost_agent <= 2 * self.goal_range and not self.reward_grad[0]:
            self.reward_grad[0] = 1
            reward += 64
        elif max_cost_agent <= 4 * self.goal_range and not self.reward_grad[1]:
            self.reward_grad[1] = 1
            reward += 64
        elif max_cost_agent <= 6 * self.goal_range and not self.reward_grad[2]:
            self.reward_grad[2] = 1
            reward += 32
        elif max_cost_agent <= 8 * self.goal_range and not self.reward_grad[3]:
            self.reward_grad[3] = 1
            reward += 32
        elif max_cost_agent <= 10 * self.goal_range and not self.reward_grad[4]:
            self.reward_grad[4] = 1
            reward += 32
        elif max_cost_agent <= 12 * self.goal_range and not self.reward_grad[5]:
            self.reward_grad[5] = 1
            reward += 16
        elif max_cost_agent <= 14 * self.goal_range and not self.reward_grad[6]:
            self.reward_grad[6] = 1
            reward += 16
        elif max_cost_agent <= 16 * self.goal_range and not self.reward_grad[7]:
            self.reward_grad[7] = 1
            reward += 16
        elif max_cost_agent <= 18 * self.goal_range and not self.reward_grad[8]:
            self.reward_grad[8] = 1
            reward += 8
        elif max_cost_agent <= 20 * self.goal_range and not self.reward_grad[9]:
          self.reward_grad[9] = 1
          reward += 8
        elif max_cost_agent <= 22 * self.goal_range and not self.reward_grad[10]:
          self.reward_grad[10] = 1
          reward += 8
        elif max_cost_agent <= 24 * self.goal_range and not self.reward_grad[11]:
          self.reward_grad[11] = 1
          reward += 4
        elif max_cost_agent <= 26 * self.goal_range and not self.reward_grad[12]:
          self.reward_grad[12] = 1
          reward += 4                    
        elif max_cost_agent <= 28 * self.goal_range and not self.reward_grad[13]:
          self.reward_grad[13] = 1
          reward += 4   

        if cost_to_go <= self.goal_range * self.robot_num and not self.reward_grad[20]:
            self.reward_grad[20] = 1
            reward += 16
        elif cost_to_go <= 2 * self.goal_range * self.robot_num and not self.reward_grad[21]:
            self.reward_grad[21] = 1
            reward += 16
        elif cost_to_go <= 4 * self.goal_range * self.robot_num and not self.reward_grad[22]:
            self.reward_grad[22] = 1
            reward += 16
        elif cost_to_go <= 8 * self.goal_range * self.robot_num and not self.reward_grad[23]:
            self.reward_grad[23] = 1
            reward += 16
        elif cost_to_go <= 10 * self.goal_range * self.robot_num and not self.reward_grad[24]:
            self.reward_grad[24] = 1
            reward += 8
        elif cost_to_go <= 12 * self.goal_range * self.robot_num and not self.reward_grad[25]:
            self.reward_grad[25] = 1
            reward += 8
        elif cost_to_go <= 14 * self.goal_range * self.robot_num and not self.reward_grad[26]:
            self.reward_grad[26] = 1
            reward += 8
        elif cost_to_go <= 16 * self.goal_range * self.robot_num and not self.reward_grad[27]:
            self.reward_grad[27] = 1
            reward += 8
        elif cost_to_go <= 18 * self.goal_range * self.robot_num and not self.reward_grad[28]:
            self.reward_grad[28] = 1
            reward += 4
        elif cost_to_go <= 21 * self.goal_range * self.robot_num and not self.reward_grad[29]:
            self.reward_grad[29] = 1
            reward += 4
        elif cost_to_go <= 24 * self.goal_range * self.robot_num and not self.reward_grad[30]:
            self.reward_grad[30] = 1
            reward += 4
        elif cost_to_go <= 28 * self.goal_range * self.robot_num and not self.reward_grad[31]:
            self.reward_grad[31] = 1
            reward += 4                                    
        return done, reward

    def render(self, mode = 'human'):
        plt.gcf().clear()

        render_image = np.copy(0*self.maze).astype(np.int16)
        for i in range(self.robot_num):
            render_image[self.loc[i,0]-1:self.loc[i,0]+2, self.loc[i,1]-1:self.loc[i,1]+2] += robot_marker

        row, col = np.nonzero(render_image)
        min_robots = 150.
        max_robots = float(np.max(render_image))
        rgb_render_image = np.stack((render_image+self.maze*255,)*3, -1)

        for i in range(row.shape[0]):
            value = render_image[row[i],col[i]]
            ratio = 0.4 + 0.5 * max(value - min_robots, 0) / (max_robots - min_robots)
            ratio = min(0.9, max(0.4, ratio))
            b = 255
            g = 255 * (1 - ratio)
            r = 255 * (1 - ratio)

            for j, rgb in enumerate([r,g,b]):
                rgb_render_image[row[i], col[i], j] = np.uint8(rgb)

        plt.imshow(rgb_render_image.astype(np.uint8), vmin=0, vmax=255)
        plt.show(False)
        plt.pause(0.0001)

    def reset(self):
        return self._build_robot()

    def transfer_learning(self):
        if not self.tflearn:
            self.tflearn = True
            self.cur_robot = np.argmax(costData[self.loc[:, 0], self.loc[:, 1]])

    def instructor(self):

        _cost_to_goal = costData[self.loc[self.cur_robot, 0], self.loc[self.cur_robot, 1]]
        if _cost_to_goal < 5:
          self.cur_robot = np.argmax(costData[self.loc[:, 0], self.loc[:, 1]])
          _cost_to_goal = costData[self.loc[self.cur_robot, 0], self.loc[self.cur_robot, 1]]

        for i in range(-1, 2):
          for j in range(-1, 2):
            if not (i == 0 and j == 0):
              new_pt = self.loc[self.cur_robot, :] + np.array([i, j]).astype(int)
              new_cost = costData[new_pt[0], new_pt[1]]
              if new_cost > 0 and new_cost < _cost_to_goal:
                action = self.rev_action_map.get((i, j))
                _cost_to_goal = np.copy(new_cost)

        return action

    def expert(self, robot_id):

        done, _ = self.get_reward()

        if robot_id is None or done:
            return self.expert_restart_session()

        _cost_to_goal = costData[self.loc[robot_id,0], self.loc[robot_id,1]]
        if _cost_to_goal >1:

            for i in range(-1,2):
                for j in range(-1,2):
                    if not (i==0 and j==0):
                        new_pt = self.loc[robot_id, :] + np.array([i,j]).astype(int)
                        new_cost = costData[new_pt[0], new_pt[1]]
                        if new_cost>0 and new_cost < _cost_to_goal:
                            action = self.rev_action_map.get((i,j))
                            _cost_to_goal = np.copy(new_cost)

            return action, robot_id

        else:
            return self.expert_restart_session()

    def expert_restart_session(self):
        done, _ = self.get_reward()
        if done:
            self.reset()
        robot_id = np.argmax(costData[self.loc[:, 0], self.loc[:, 1]])
        return self.expert(robot_id)

def main(MazeEnv):
    env = MazeEnv()
    env.render()
    plt.pause(2)
    n_epochs = 10000
    robot_id = None
    steps  = 0
    rewards = 0.0
    import time
    start = time.time()
    for i in range(n_epochs):
        if i%50 ==0:
            now = time.time()
            print('step {} time elapse {}'.format(i, now-start))
            start = now
        steps+=1
        #next_action = np.random.randint(4,size = 1)
        next_action, robot_id = env.expert(robot_id)
        state_img,reward, done, _ = env.step(next_action)
        rewards +=reward
        env.render()
        print('Step = %d, rewards = %.1f, reward = %.1f, done = %d'%(steps, rewards, reward, done), end='\r')
        if done:
            print('\n')


        if done:
            steps = 0
            rewards = 0.0
            plt.pause(2)
            env.reset()


if __name__ == '__main__':
    main(Maze0122Env)


