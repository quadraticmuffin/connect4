from abc import ABC, abstractmethod
import numpy as np
import torch as th
from torch import nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3.common.distributions import (
    Distribution,
)

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
class MaskPolicy(BasePolicy, ABC):
    @abstractmethod
    def _predict(self, observation: PyTorchObs, action_mask: np.ndarray, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def predict(
        self,
        observation: PyTorchObs,
        action_mask: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Copied straight from BasePolicy except the _predict call takes action_mask
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, action_mask, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]

class RandomPolicy(MaskPolicy):
    def _predict(self, state, action_mask, deterministic=False):
        assert not deterministic
        return th.as_tensor(np.random.choice(np.nonzero(action_mask)[0]), device=self.device)
    
class CNNExtractor(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "CNNExtractor must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    
class MaskedACPolicy(MaskPolicy, ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, features_extractor_class=CNNExtractor, **kwargs)
    
    def forward(self, obs: th.Tensor, action_mask: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, action_mask: th.Tensor
    ) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        Masks invalid actions if the action distribution is Categorical

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        # Here latent_pi are the logits before the softmax
        if action_mask is not None:
            invalid_actions = th.tensor(-float("inf")).to(
                dtype=mean_actions.dtype, device=mean_actions.device
            )
            mean_actions = th.where(action_mask, mean_actions, invalid_actions)
            # print action probabilities for debugging
            # if self.print_action_probs:
            #     action_mask = action_mask.unsqueeze(0)
            #     probs = logits_to_probs(mean_actions)[action_mask]
            #     actions = ACTIONS[action_mask]
            #     for i in range(len(probs)):
            #         print(f"{actions[i]}: {probs[i]}")
            #     print()

            if len(action_mask.shape) > 1 and sum(action_mask.sum(dim=1) == 0): 
                raise ValueError(f'mask is all zeros at index {(action_mask.sum(dim=1) == 0).nonzero()}')
        out = self.action_dist.proba_distribution(action_logits=mean_actions)
        return out

    def _predict(self, observation: PyTorchObs, action_mask: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation, action_mask).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, action_mask: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def get_distribution(self, obs: PyTorchObs, action_mask: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, action_mask)
    
    # def predict(
    #     self,
    #     observation: Union[np.ndarray, Dict[str, np.ndarray]],
    #     state: Optional[Tuple[np.ndarray, ...]] = None,
    #     episode_start: Optional[np.ndarray] = None,
    #     deterministic: bool = False,
    # ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
    #     """
    #     Get the policy action from an observation (and optional hidden state).
    #     Includes sugar-coating to handle different observations (e.g. normalizing images).

    #     :param observation: the input observation
    #     :param state: The last hidden states (can be None, used in recurrent policies)
    #     :param episode_start: The last masks (can be None, used in recurrent policies)
    #         this correspond to beginning of episodes,
    #         where the hidden states of the RNN must be reset.
    #     :param deterministic: Whether or not to return deterministic actions.
    #     :return: the model's action and the next hidden state
    #         (used in recurrent policies)
    #     """
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.set_training_mode(False)

    #     # Check for common mistake that the user does not mix Gym/VecEnv API
    #     # Tuple obs are not supported by SB3, so we can safely do that check
    #     if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
    #         raise ValueError(
    #             "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
    #             "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
    #             "vs `obs = vec_env.reset()` (SB3 VecEnv). "
    #             "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
    #             "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
    #         )

    #     obs_tensor, vectorized_env = self.obs_to_tensor(observation)

    #     with th.no_grad():
    #         actions = self._predict(obs_tensor, action_mask, deterministic=deterministic)
    #     # Convert to numpy, and reshape to the original action shape
    #     actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

    #     if isinstance(self.action_space, spaces.Box):
    #         if self.squash_output:
    #             # Rescale to proper domain when using squashing
    #             actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
    #         else:
    #             # Actions could be on arbitrary scale, so clip the actions to avoid
    #             # out of bound error (e.g. if sampling from a Gaussian distribution)
    #             actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

    #     # Remove batch dimension if needed
    #     if not vectorized_env:
    #         assert isinstance(actions, np.ndarray)
    #         actions = actions.squeeze(axis=0)

    #     return actions, state  # type: ignore[return-value]