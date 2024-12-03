import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import Connect4
from policies import MaskPolicy, RandomPolicy

class Connect4Env(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, opp_policy: MaskPolicy = None):
        super().__init__()
        self.game = Connect4()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(4, 6, 7), dtype=np.uint8)
        
        if opp_policy is None:
            self.opp_policy = RandomPolicy(self.observation_space, self.action_space)
        else:
            self.opp_policy = opp_policy

    def get_state(self):
        board = self.game.board
        return np.stack((board!=0, board==1, board==2, np.full_like(board, self.game.current_player==2)))
    
    def step(self, action):
        won = self.game.move(action)
        opp_won = False
        if not won and not self.board_full():
            # opponent makes their move
            opp_move, _ = self.opp_policy.predict(self.get_state(), self.action_mask())
            assert opp_move in np.nonzero(self.action_mask())[0]
            opp_won = self.game.move(opp_move)
        done = won or opp_won or self.board_full()
        reward = int(won) - int(opp_won)
        observation = self.get_state()
        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.game = Connect4()
        return self.get_state(), {}

    def render(self):
        print("current player:", self.game.current_player)
        for row in self.game.board:
            print(row)

    def close(self):
        pass

    def action_mask(self):
        return self.game.board[0]==0

    def board_full(self):
        return self.action_mask().sum() == 0



if __name__ == "__main__":
    env = Connect4Env()
    print(env.get_state())
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(1)
    print(env.get_state())
