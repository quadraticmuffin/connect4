from mask_ppo import MaskPPO
from environment import Connect4Env
from policies import MaskedACPolicy

lr_schedule = lambda x: 3e-4 / (8 * (1 - x) + 1)**1
model = MaskPPO(
    MaskedACPolicy, 
    Connect4Env(), 
    learning_rate=lr_schedule,
    verbose=1,
)
model.learn(total_timesteps=100000)