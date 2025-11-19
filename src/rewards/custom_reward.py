"""
Custom reward function for Rocket League bot training
"""
import numpy as np
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK


class CustomReward(RewardFunction):
    """
    A baseline reward function that encourages:
    - Moving towards the ball
    - Touching the ball
    - Scoring goals
    - Moving the ball towards opponent's goal
    - Ball velocity towards opponent's goal
    """

    def __init__(
        self,
        ball_touch_reward=1.0,
        goal_reward=10.0,
        velocity_ball_to_goal_weight=0.5,
        velocity_player_to_ball_weight=0.3,
        distance_to_ball_weight=0.1,
    ):
        super().__init__()
        self.ball_touch_reward = ball_touch_reward
        self.goal_reward = goal_reward
        self.velocity_ball_to_goal_weight = velocity_ball_to_goal_weight
        self.velocity_player_to_ball_weight = velocity_player_to_ball_weight
        self.distance_to_ball_weight = distance_to_ball_weight

        # Track previous ball touch for each player
        self.last_touched = {}

    def reset(self, initial_state: GameState):
        """Reset tracking variables"""
        self.last_touched = {player.car_id: False for player in initial_state.players}

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """Calculate reward for a single player"""
        reward = 0.0

        # Get ball position and velocity
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        # Get player position and velocity
        player_pos = player.car_data.position
        player_vel = player.car_data.linear_velocity

        # Determine opponent's goal position
        if player.team_num == BLUE_TEAM:
            opponent_goal = np.array(ORANGE_GOAL_BACK)
        else:
            opponent_goal = np.array(BLUE_GOAL_BACK)

        # 1. Reward for touching the ball
        if player.ball_touched:
            if not self.last_touched.get(player.car_id, False):
                reward += self.ball_touch_reward
            self.last_touched[player.car_id] = True
        else:
            self.last_touched[player.car_id] = False

        # 2. Distance to ball (closer is better)
        distance_to_ball = np.linalg.norm(ball_pos - player_pos)
        # Normalize and invert (closer = higher reward)
        distance_reward = np.exp(-distance_to_ball / 2000)  # 2000 is arbitrary scaling
        reward += distance_reward * self.distance_to_ball_weight

        # 3. Velocity towards ball
        ball_direction = ball_pos - player_pos
        ball_direction_norm = ball_direction / (np.linalg.norm(ball_direction) + 1e-5)
        velocity_to_ball = np.dot(player_vel, ball_direction_norm)
        # Normalize to [-1, 1] range
        velocity_to_ball_norm = np.clip(velocity_to_ball / 2300, -1, 1)  # 2300 is max car speed
        reward += velocity_to_ball_norm * self.velocity_player_to_ball_weight

        # 4. Ball velocity towards opponent goal
        goal_direction = opponent_goal - ball_pos
        goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-5)
        ball_vel_to_goal = np.dot(ball_vel, goal_direction_norm)
        # Normalize to [-1, 1] range
        ball_vel_to_goal_norm = np.clip(ball_vel_to_goal / 6000, -1, 1)  # 6000 is max ball speed
        reward += ball_vel_to_goal_norm * self.velocity_ball_to_goal_weight

        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """Calculate reward at episode end (e.g., goal scored)"""
        reward = 0.0

        # Check if player's team scored
        if state.blue_score > 0 and player.team_num == BLUE_TEAM:
            reward += self.goal_reward
        elif state.orange_score > 0 and player.team_num == ORANGE_TEAM:
            reward += self.goal_reward
        # Negative reward for conceding
        elif state.blue_score > 0 and player.team_num == ORANGE_TEAM:
            reward -= self.goal_reward
        elif state.orange_score > 0 and player.team_num == BLUE_TEAM:
            reward -= self.goal_reward

        return reward
