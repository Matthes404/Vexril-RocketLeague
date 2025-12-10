"""
Gymnasium wrapper for RLGym-Sim environment compatibility with Stable-Baselines3
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OpponentPolicy:
    """
    Base class for opponent policies during training.
    Subclass this to create different opponent behaviors.
    """

    def get_action(self, obs) -> np.ndarray:
        """Return an 8D action array for the opponent"""
        raise NotImplementedError


class ZeroOpponent(OpponentPolicy):
    """Opponent that does nothing (original behavior - NOT recommended)"""

    def get_action(self, obs) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


class RandomOpponent(OpponentPolicy):
    """Opponent that takes random actions (basic exploration)"""

    def get_action(self, obs) -> np.ndarray:
        return np.random.uniform(-1, 1, size=8).astype(np.float32)


class BasicChaseOpponent(OpponentPolicy):
    """
    Simple opponent that tries to drive towards the ball.
    This provides basic opposition without being too sophisticated.
    """

    def get_action(self, obs) -> np.ndarray:
        # Create action array
        action = np.zeros(8, dtype=np.float32)

        # Always drive forward
        action[0] = 1.0  # throttle

        # Add some random steering for variety
        action[1] = np.random.uniform(-0.3, 0.3)

        # Occasionally boost
        if np.random.random() < 0.3:
            action[6] = 1.0  # boost

        # Occasionally jump (for aerial attempts)
        if np.random.random() < 0.05:
            action[5] = 1.0  # jump

        return action


class MixedOpponent(OpponentPolicy):
    """
    Opponent that switches between different behaviors.
    Provides varied opposition during training.
    """

    def __init__(self):
        self.policies = [RandomOpponent(), BasicChaseOpponent(), ZeroOpponent()]
        self.current_policy = 0
        self.steps = 0
        self.switch_interval = 500  # Switch policy every N steps

    def get_action(self, obs) -> np.ndarray:
        self.steps += 1
        if self.steps >= self.switch_interval:
            self.steps = 0
            self.current_policy = np.random.randint(0, len(self.policies))
        return self.policies[self.current_policy].get_action(obs)


class RLGymSimWrapper(gym.Env):
    """
    Wrapper to make RLGym-Sim environment compatible with SB3.

    In self-play mode (default for multi-agent), this wrapper:
    - Uses the SAME model to control ALL agents
    - Returns observations/rewards from ALL agents (alternating each step)
    - Effectively doubles training data for 1v1, triples for 1v1v1, etc.

    This ensures we don't waste 50% of training data in self-play!
    """

    def __init__(self, rlgym_env, opponent_policy: str = "self", self_play: bool = True):
        """
        Initialize wrapper

        Args:
            rlgym_env: RLGym-Sim environment instance
            opponent_policy: Type of opponent policy to use during training.
                Options: "self" (self-play, recommended), "zero", "random", "chase", "mixed"
                In self-play mode, all agents use the model's actions.
            self_play: If True (default), use self-play where all agents contribute training data.
                       If False, only agent 0's data is used (wastes 50% of data in 1v1).
        """
        super().__init__()
        self.env = rlgym_env
        self.self_play = self_play

        # Initialize opponent policy (only used if self_play=False)
        opponent_policies = {
            "self": None,  # Self-play: no scripted opponent
            "zero": ZeroOpponent,
            "random": RandomOpponent,
            "chase": BasicChaseOpponent,
            "mixed": MixedOpponent
        }

        if opponent_policy == "self" or self_play:
            self.opponent = None
            self.self_play = True
            print("Training mode: SELF-PLAY (all agents use model, all data used for training)")
        else:
            if opponent_policy not in opponent_policies:
                print(f"WARNING: Unknown opponent policy '{opponent_policy}', using 'mixed'")
                opponent_policy = "mixed"
            self.opponent = opponent_policies[opponent_policy]()
            print(f"Training opponent policy: {opponent_policy}")

        # Get a sample observation to determine the space and number of agents
        sample_obs = self.env.reset()

        # Determine number of agents from observation structure
        # RLGym-Sim returns a list or array of observations when there are multiple agents
        if isinstance(sample_obs, (list, np.ndarray)) and hasattr(sample_obs, '__len__'):
            # Check if it's a multi-agent observation (list of arrays or 2D array)
            if isinstance(sample_obs, list):
                self.num_agents = len(sample_obs)
                sample_to_use = sample_obs[0]
            elif isinstance(sample_obs, np.ndarray) and sample_obs.ndim == 2:
                # 2D array: (num_agents, obs_dim)
                self.num_agents = sample_obs.shape[0]
                sample_to_use = sample_obs[0]
            else:
                # 1D array: single agent observation
                self.num_agents = 1
                sample_to_use = sample_obs
        else:
            self.num_agents = 1
            sample_to_use = sample_obs

        print(f"Detected {self.num_agents} agents, sample observation shape: {np.array(sample_to_use).shape}")

        # For self-play: track which agent's perspective we're returning
        # We alternate between agents to use ALL training data
        self.current_agent_idx = 0

        # Store all agents' observations for self-play
        self._all_obs = None
        self._all_rewards = None
        self._pending_obs = []  # Queue of observations to return

        # Flatten observation if needed
        if isinstance(sample_to_use, dict):
            # If obs is a dict, we need to flatten it
            sample_flat = self._flatten_obs(sample_to_use)
        elif isinstance(sample_to_use, (list, tuple)) and not isinstance(sample_obs, list):
            # If it's a list/tuple (but not multi-agent), convert to array
            sample_flat = np.array(sample_to_use).flatten()
        else:
            # Already an array
            sample_flat = np.array(sample_to_use).flatten()

        # Define observation space
        obs_dim = sample_flat.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Get action space from env and convert to gymnasium space
        if hasattr(self.env, 'action_space'):
            # Convert gym.spaces to gymnasium.spaces
            env_action_space = self.env.action_space
            # Create a new gymnasium Box space with the same parameters
            self.action_space = spaces.Box(
                low=env_action_space.low,
                high=env_action_space.high,
                shape=env_action_space.shape,
                dtype=env_action_space.dtype
            )
        else:
            # Default action space for RLGym (8-dimensional continuous)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32
            )

    def _flatten_obs(self, obs):
        """Flatten observation to 1D array"""
        if isinstance(obs, dict):
            # Concatenate all dict values
            return np.concatenate([np.array(v).flatten() for v in obs.values()])
        elif isinstance(obs, (list, tuple)):
            return np.array(obs).flatten()
        else:
            return np.array(obs).flatten()

    def _extract_agent_obs(self, obs, agent_idx):
        """Extract a single agent's observation from multi-agent obs"""
        if isinstance(obs, list):
            return obs[agent_idx]
        elif isinstance(obs, np.ndarray) and obs.ndim == 2:
            return obs[agent_idx]
        else:
            return obs

    def _extract_agent_value(self, value, agent_idx):
        """Extract a single agent's value (reward/done) from multi-agent value"""
        if isinstance(value, (list, np.ndarray)) and hasattr(value, '__getitem__'):
            try:
                return value[agent_idx]
            except (IndexError, TypeError):
                return value
        return value

    def reset(self, seed=None, options=None):
        """Reset environment"""
        # RLGym-Sim reset doesn't take seed parameter
        obs = self.env.reset()

        # Store all observations
        self._all_obs = obs

        if self.self_play and self.num_agents > 1:
            # In self-play, alternate which agent's perspective we start from
            # This ensures balanced training across all agents
            agent_obs = self._extract_agent_obs(obs, self.current_agent_idx)
        else:
            # Non-self-play: just use first agent
            self.current_agent_idx = 0
            agent_obs = self._extract_agent_obs(obs, 0) if self.num_agents > 1 else obs

        obs_flat = self._flatten_obs(agent_obs).astype(np.float32)

        # Gymnasium API requires (obs, info) tuple
        return obs_flat, {}

    def step(self, action):
        """
        Step environment.

        In self-play mode:
        - All agents use the model's action (adapted for their perspective)
        - We alternate which agent's data we return to SB3
        - This uses ALL training data from all agents!
        """
        # Convert action to numpy array if needed
        action = np.array(action, dtype=np.float32)

        if self.self_play and self.num_agents > 1:
            # Self-play mode: use same action for all agents, alternate data returned
            # All agents are the same model, so they all use the same action logic
            actions = np.array([action for _ in range(self.num_agents)])

            # Step environment with all actions
            obs, reward, done, info = self.env.step(actions)

            # Store all observations for next step
            self._all_obs = obs

            # Get data for current agent
            agent_obs = self._extract_agent_obs(obs, self.current_agent_idx)
            agent_reward = self._extract_agent_value(reward, self.current_agent_idx)
            agent_done = self._extract_agent_value(done, self.current_agent_idx)

            # Alternate which agent we return data for
            # This ensures we use training data from ALL agents
            self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents

            obs_flat = self._flatten_obs(agent_obs).astype(np.float32)
            return obs_flat, float(agent_reward), bool(agent_done), False, info

        else:
            # Non-self-play mode: use scripted opponent
            if self.num_agents > 1:
                # Create actions for all agents
                actions = [action]
                # Add opponent actions
                for _ in range(self.num_agents - 1):
                    opponent_action = self.opponent.get_action(None)
                    actions.append(opponent_action)
                action = np.array(actions)

            # RLGym returns (obs, reward, done, info)
            obs, reward, done, info = self.env.step(action)

            # Store for next step
            self._all_obs = obs

            # Handle multi-agent observations (extract first agent's obs)
            agent_obs = self._extract_agent_obs(obs, 0) if self.num_agents > 1 else obs
            agent_reward = self._extract_agent_value(reward, 0) if self.num_agents > 1 else reward
            agent_done = self._extract_agent_value(done, 0) if self.num_agents > 1 else done

            # Flatten observation
            obs_flat = self._flatten_obs(agent_obs).astype(np.float32)

            return obs_flat, float(agent_reward), bool(agent_done), False, info

    def render(self):
        """Render environment"""
        if hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        """Close environment"""
        if hasattr(self.env, 'close'):
            self.env.close()
