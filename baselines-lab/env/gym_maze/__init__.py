from gym.envs.registration import register

register(id="Maze0318-v0",
        entry_point="env.gym_maze.envs:MazeBase",
        max_episode_steps=2000,
        kwargs={'map_file':'../mapdata/map0318.csv', 'goal':[82, 80], 'goal_range':10})