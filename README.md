# connect4

Quick half-day hack to remind myself of some RL concepts. Trains an actor-critic network to play Connect 4.
Main meat is a clean-ish augmentation of [PPO](https://arxiv.org/abs/1707.06347) from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo).
The augmentation is to include [invalid action masking](https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html), and can be found in [mask_ppo.py](mask_ppo.py) and [policies.py](policies.py).

Currently trains as Player 1 against a random player, and is pretty quickly able to reach a good winrate against it.
To add self-play, one would want to create a callback function which periodically would replace the opponent with the most recent (or so-far best) policy.

Would be curious to see how competitive a neural net can possibly get vs. [the optimal solution](http://blog.gamesolver.org/solving-connect-four/01-introduction/).
