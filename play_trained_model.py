import numpy as np
import torch as th
from mask_ppo import MaskPPO
from environment import Connect4Env, MaskPolicy
from game import Connect4
from stable_baselines3.common.type_aliases import PyTorchObs

class ManualPolicy(MaskPolicy):
    def _predict(self, observation: PyTorchObs, action_mask: np.ndarray, deterministic: bool = False) -> th.Tensor:
        valid_actions = np.nonzero(action_mask)[0]
        

env = Connect4Env(ManualPolicy(None, None))

model = MaskPPO.load(
    'model.zip',
    None,
    device='cpu',
).policy

game = Connect4()
print(game.board)

while True:
    board = game.board
    state = np.stack((board!=0, board==1, board==2, np.full_like(board, game.current_player==2)))
    mask = board[0]==0
    bot_move, _ = model.predict(th.as_tensor(state), th.as_tensor(mask))
    print('bot move:', bot_move)
    bot_won = game.move(bot_move.item())
    game.display()
    if bot_won:
        print('bot won')
        break
    elif sum(board[0]==0) == 0:
        print('tied')
        break
    while True:
        try:
            usr_in = input('choose action: ')
            if usr_in=='EXIT': break
            action = int(usr_in)
        except:
            continue
        if action >=0 and action < game.board.shape[1] and game.board[0,action] == 0:
            break
        print(f'invalid action. must be a valid int 0-6')
    you_won = game.move(action)
    game.display()
    if you_won:
        print('you won')
        break
    elif sum(board[0]==0) == 0:
        print('tied')
        break
    