"""
Comprehensive reward function for Rocket League bot training

This reward function implements advanced RL concepts and Rocket League-specific shaping:
- Dense vs sparse rewards with proper balance
- Ball velocity change tracking and impact-based touch rewards
- Car-ball and car-goal alignment rewards
- Movement quality (forward velocity, uprightness, boost management)
- Positioning intelligence (goal-side, shadow defense)
- Shot/clear/deflection detection
- Proper normalization, clipping, and temporal smoothing
"""
import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, BALL_RADIUS, CAR_MAX_SPEED, SUPERSONIC_THRESHOLD, BALL_MAX_SPEED
from typing import Dict, Tuple


class CustomReward(RewardFunction):
    """
    Comprehensive reward function with multiple components for training intelligent RL agents
    """

    def __init__(
        self,
        # Sparse rewards
        goal_reward=10.0,
        concede_penalty=-10.0,

        # Touch-based rewards (semi-sparse)
        touch_ball_reward=0.5,
        touch_impact_weight=2.0,  # Reward based on ball velocity change from touch
        shot_reward=3.0,  # Successful shot towards goal
        clear_reward=2.0,  # Successful clear away from own goal
        deflection_reward=1.5,  # Successful deflection

        # Alignment rewards (dense)
        car_ball_alignment_weight=0.3,
        car_goal_alignment_weight=0.2,

        # Ball control rewards (dense)
        ball_velocity_to_goal_weight=0.5,
        ball_line_prediction_weight=0.3,  # Reward being on ball's trajectory

        # Movement rewards (dense)
        forward_velocity_weight=0.2,
        uprightness_weight=0.1,
        reverse_penalty_weight=-0.15,
        boost_efficiency_weight=0.1,
        flip_reward=0.3,

        # Positioning rewards (dense)
        goal_side_positioning_weight=0.4,
        shadow_defense_weight=0.3,
        distance_to_ball_weight=0.1,

        # Training stability
        reward_clip_range=10.0,
        temporal_smoothing_alpha=0.3,  # EMA smoothing factor
        normalize_rewards=True,
    ):
        super().__init__()

        # Store all weights
        # Sparse rewards
        self.goal_reward = goal_reward
        self.concede_penalty = concede_penalty

        # Touch-based rewards
        self.touch_ball_reward = touch_ball_reward
        self.touch_impact_weight = touch_impact_weight
        self.shot_reward = shot_reward
        self.clear_reward = clear_reward
        self.deflection_reward = deflection_reward

        # Alignment rewards
        self.car_ball_alignment_weight = car_ball_alignment_weight
        self.car_goal_alignment_weight = car_goal_alignment_weight

        # Ball control rewards
        self.ball_velocity_to_goal_weight = ball_velocity_to_goal_weight
        self.ball_line_prediction_weight = ball_line_prediction_weight

        # Movement rewards
        self.forward_velocity_weight = forward_velocity_weight
        self.uprightness_weight = uprightness_weight
        self.reverse_penalty_weight = reverse_penalty_weight
        self.boost_efficiency_weight = boost_efficiency_weight
        self.flip_reward = flip_reward

        # Positioning rewards
        self.goal_side_positioning_weight = goal_side_positioning_weight
        self.shadow_defense_weight = shadow_defense_weight
        self.distance_to_ball_weight = distance_to_ball_weight

        # Training stability
        self.reward_clip_range = reward_clip_range
        self.temporal_smoothing_alpha = temporal_smoothing_alpha
        self.normalize_rewards = normalize_rewards

        # Tracking variables
        self.last_touched = {}
        self.last_ball_velocity = None
        self.last_ball_position = None
        self.previous_rewards = {}  # For temporal smoothing
        self.touch_timer = {}  # Track time since last touch
        self.last_boost_amount = {}
        self.last_on_ground = {}
        self.last_player_velocity = {}

    def reset(self, initial_state: GameState):
        """Reset all tracking variables"""
        self.last_touched = {player.car_id: False for player in initial_state.players}
        self.last_ball_velocity = initial_state.ball.linear_velocity.copy()
        self.last_ball_position = initial_state.ball.position.copy()
        self.previous_rewards = {player.car_id: 0.0 for player in initial_state.players}
        self.touch_timer = {player.car_id: 0 for player in initial_state.players}
        self.last_boost_amount = {player.car_id: player.boost_amount for player in initial_state.players}
        self.last_on_ground = {player.car_id: player.on_ground for player in initial_state.players}
        self.last_player_velocity = {player.car_id: player.car_data.linear_velocity.copy() for player in initial_state.players}

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """Calculate comprehensive reward for a single player"""
        reward = 0.0

        # Get basic data
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        player_pos = player.car_data.position
        player_vel = player.car_data.linear_velocity
        player_forward = player.car_data.forward()
        player_up = player.car_data.up()

        # Determine goals
        if player.team_num == BLUE_TEAM:
            opponent_goal = np.array(ORANGE_GOAL_BACK)
            own_goal = np.array(BLUE_GOAL_BACK)
        else:
            opponent_goal = np.array(BLUE_GOAL_BACK)
            own_goal = np.array(ORANGE_GOAL_BACK)

        # ========== TOUCH-BASED REWARDS (SEMI-SPARSE) ==========

        ball_velocity_change = 0.0
        if self.last_ball_velocity is not None:
            ball_velocity_change = np.linalg.norm(ball_vel - self.last_ball_velocity)

        if player.ball_touched:
            if not self.last_touched.get(player.car_id, False):
                # Base touch reward
                reward += self.touch_ball_reward

                # Impact-based touch reward (based on ball velocity change)
                impact_reward = (ball_velocity_change / BALL_MAX_SPEED) * self.touch_impact_weight
                reward += impact_reward

                # Shot detection: Ball moving towards opponent goal after touch
                shot_reward = self._calculate_shot_reward(ball_pos, ball_vel, opponent_goal)
                reward += shot_reward

                # Clear detection: Ball moving away from own goal after touch
                clear_reward = self._calculate_clear_reward(ball_pos, ball_vel, own_goal)
                reward += clear_reward

                # Deflection reward: Significant change in ball direction
                deflection_reward = self._calculate_deflection_reward(ball_velocity_change)
                reward += deflection_reward

                # Reset touch timer
                self.touch_timer[player.car_id] = 0

            self.last_touched[player.car_id] = True
        else:
            self.last_touched[player.car_id] = False
            self.touch_timer[player.car_id] = self.touch_timer.get(player.car_id, 0) + 1

        # ========== ALIGNMENT REWARDS (DENSE) ==========

        # Car-ball alignment: Reward facing towards ball
        car_ball_alignment = self._calculate_alignment_reward(
            player_pos, ball_pos, player_forward
        )
        reward += car_ball_alignment * self.car_ball_alignment_weight

        # Car-goal alignment: Reward facing towards opponent goal (when close to ball)
        distance_to_ball = np.linalg.norm(ball_pos - player_pos)
        if distance_to_ball < 1500:  # Only reward goal alignment when close to ball
            car_goal_alignment = self._calculate_alignment_reward(
                player_pos, opponent_goal, player_forward
            )
            reward += car_goal_alignment * self.car_goal_alignment_weight

        # ========== BALL CONTROL REWARDS (DENSE) ==========

        # Ball velocity towards opponent goal
        ball_vel_to_goal = self._calculate_velocity_to_target(
            ball_vel, ball_pos, opponent_goal
        )
        reward += ball_vel_to_goal * self.ball_velocity_to_goal_weight

        # Ball line prediction: Reward being on ball's trajectory
        ball_line_reward = self._calculate_ball_line_reward(
            player_pos, ball_pos, ball_vel
        )
        reward += ball_line_reward * self.ball_line_prediction_weight

        # ========== MOVEMENT REWARDS (DENSE) ==========

        # Forward velocity reward: Encourage moving forward fast
        forward_vel_reward = self._calculate_forward_velocity_reward(
            player_vel, player_forward
        )
        reward += forward_vel_reward * self.forward_velocity_weight

        # Uprightness reward: Encourage staying upright (wheels on ground orientation)
        uprightness_reward = self._calculate_uprightness_reward(player_up)
        reward += uprightness_reward * self.uprightness_weight

        # Reverse penalty: Discourage driving backwards
        reverse_penalty = self._calculate_reverse_penalty(player_vel, player_forward)
        reward += reverse_penalty * self.reverse_penalty_weight

        # Boost efficiency: Reward using boost when moving forward at high speed
        boost_efficiency = self._calculate_boost_efficiency(
            player, player_vel, player_forward
        )
        reward += boost_efficiency * self.boost_efficiency_weight

        # Flip detection reward: Encourage using flips for recovery and speed
        flip_reward = self._detect_flip(player)
        reward += flip_reward * self.flip_reward

        # ========== POSITIONING REWARDS (DENSE) ==========

        # Goal-side positioning: Encourage being between ball and own goal
        goal_side_reward = self._calculate_goal_side_positioning(
            player_pos, ball_pos, own_goal
        )
        reward += goal_side_reward * self.goal_side_positioning_weight

        # Shadow defense: Reward defensive positioning when ball is on opponent's side
        shadow_defense_reward = self._calculate_shadow_defense(
            player_pos, ball_pos, own_goal, opponent_goal
        )
        reward += shadow_defense_reward * self.shadow_defense_weight

        # Distance to ball: Encourage being closer to ball (exponential decay)
        distance_reward = np.exp(-distance_to_ball / 3000.0)
        reward += distance_reward * self.distance_to_ball_weight

        # ========== TRAINING STABILITY ==========

        # Update tracking variables
        self.last_ball_velocity = ball_vel.copy()
        self.last_ball_position = ball_pos.copy()
        self.last_boost_amount[player.car_id] = player.boost_amount
        self.last_on_ground[player.car_id] = player.on_ground
        self.last_player_velocity[player.car_id] = player_vel.copy()

        # Temporal smoothing: Smooth rewards over time using EMA
        if self.temporal_smoothing_alpha > 0:
            prev_reward = self.previous_rewards.get(player.car_id, 0.0)
            reward = (self.temporal_smoothing_alpha * reward +
                     (1 - self.temporal_smoothing_alpha) * prev_reward)
            self.previous_rewards[player.car_id] = reward

        # Reward clipping
        if self.reward_clip_range > 0:
            reward = np.clip(reward, -self.reward_clip_range, self.reward_clip_range)

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
            reward += self.concede_penalty
        elif state.orange_score > 0 and player.team_num == BLUE_TEAM:
            reward += self.concede_penalty

        return reward

    # ========== HELPER METHODS ==========

    def _calculate_shot_reward(self, ball_pos: np.ndarray, ball_vel: np.ndarray,
                               opponent_goal: np.ndarray) -> float:
        """Detect and reward shots towards opponent goal"""
        goal_direction = opponent_goal - ball_pos
        goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-5)

        # Check if ball is moving towards goal
        ball_vel_norm = ball_vel / (np.linalg.norm(ball_vel) + 1e-5)
        alignment = np.dot(ball_vel_norm, goal_direction_norm)

        # Reward if well-aligned and fast
        if alignment > 0.7:  # Threshold for "towards goal"
            speed_factor = min(np.linalg.norm(ball_vel) / BALL_MAX_SPEED, 1.0)
            return self.shot_reward * alignment * speed_factor
        return 0.0

    def _calculate_clear_reward(self, ball_pos: np.ndarray, ball_vel: np.ndarray,
                                own_goal: np.ndarray) -> float:
        """Detect and reward clears away from own goal"""
        goal_direction = own_goal - ball_pos
        goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-5)

        # Check if ball is moving away from own goal
        ball_vel_norm = ball_vel / (np.linalg.norm(ball_vel) + 1e-5)
        alignment = np.dot(ball_vel_norm, goal_direction_norm)

        # Reward if moving away from goal (negative alignment)
        if alignment < -0.6:  # Threshold for "away from goal"
            speed_factor = min(np.linalg.norm(ball_vel) / BALL_MAX_SPEED, 1.0)
            return self.clear_reward * abs(alignment) * speed_factor
        return 0.0

    def _calculate_deflection_reward(self, ball_velocity_change: float) -> float:
        """Reward significant changes in ball velocity (deflections)"""
        if ball_velocity_change > 1000:  # Threshold for deflection
            normalized_change = min(ball_velocity_change / BALL_MAX_SPEED, 1.0)
            return self.deflection_reward * normalized_change
        return 0.0

    def _calculate_alignment_reward(self, source_pos: np.ndarray,
                                   target_pos: np.ndarray,
                                   forward_vector: np.ndarray) -> float:
        """Calculate alignment between forward vector and target direction"""
        direction = target_pos - source_pos
        direction_norm = direction / (np.linalg.norm(direction) + 1e-5)

        # Dot product gives alignment (-1 to 1)
        alignment = np.dot(forward_vector, direction_norm)

        # Normalize to [0, 1] range
        return (alignment + 1) / 2

    def _calculate_velocity_to_target(self, velocity: np.ndarray,
                                     pos: np.ndarray,
                                     target: np.ndarray) -> float:
        """Calculate normalized velocity component towards target"""
        direction = target - pos
        direction_norm = direction / (np.linalg.norm(direction) + 1e-5)

        velocity_to_target = np.dot(velocity, direction_norm)
        # Normalize to [-1, 1]
        return np.clip(velocity_to_target / BALL_MAX_SPEED, -1, 1)

    def _calculate_ball_line_reward(self, player_pos: np.ndarray,
                                   ball_pos: np.ndarray,
                                   ball_vel: np.ndarray) -> float:
        """Reward being on the ball's trajectory"""
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed < 100:  # Ball is barely moving
            return 0.0

        # Calculate point on ball's trajectory line closest to player
        ball_vel_norm = ball_vel / (ball_speed + 1e-5)
        player_to_ball = player_pos - ball_pos

        # Project player position onto ball trajectory
        projection = np.dot(player_to_ball, ball_vel_norm)

        # Only reward if player is ahead of ball (on its path)
        if projection > 0:
            closest_point = ball_pos + ball_vel_norm * projection
            distance_to_line = np.linalg.norm(player_pos - closest_point)

            # Exponential decay based on distance from trajectory
            return np.exp(-distance_to_line / 1000.0)
        return 0.0

    def _calculate_forward_velocity_reward(self, velocity: np.ndarray,
                                          forward_vector: np.ndarray) -> float:
        """Reward moving forward at high speed"""
        forward_vel = np.dot(velocity, forward_vector)

        # Normalize to [0, 1] and apply square root for smoother curve
        normalized = max(forward_vel / CAR_MAX_SPEED, 0)
        return np.sqrt(normalized)

    def _calculate_uprightness_reward(self, up_vector: np.ndarray) -> float:
        """Reward staying upright (up vector aligned with world up)"""
        world_up = np.array([0, 0, 1])
        alignment = np.dot(up_vector, world_up)

        # Normalize to [0, 1]
        return (alignment + 1) / 2

    def _calculate_reverse_penalty(self, velocity: np.ndarray,
                                  forward_vector: np.ndarray) -> float:
        """Penalize driving backwards"""
        forward_vel = np.dot(velocity, forward_vector)

        # Only penalize if moving backwards
        if forward_vel < 0:
            normalized = abs(forward_vel) / CAR_MAX_SPEED
            return -normalized
        return 0.0

    def _calculate_boost_efficiency(self, player: PlayerData,
                                   velocity: np.ndarray,
                                   forward_vector: np.ndarray) -> float:
        """Reward using boost efficiently (when moving forward fast)"""
        # Check if boost was used this step
        boost_used = (self.last_boost_amount.get(player.car_id, 100) -
                     player.boost_amount)

        if boost_used > 0:
            forward_vel = np.dot(velocity, forward_vector)
            speed = np.linalg.norm(velocity)

            # Reward using boost when:
            # 1. Moving forward
            # 2. Below supersonic speed (boost is useful)
            if forward_vel > 0 and speed < SUPERSONIC_THRESHOLD:
                speed_factor = speed / SUPERSONIC_THRESHOLD
                return speed_factor
            # Penalize using boost when already supersonic or going backwards
            else:
                return -0.5
        return 0.0

    def _detect_flip(self, player: PlayerData) -> float:
        """Detect and reward flips (for recovery and speed)"""
        # Check for significant change in on_ground status (aerial -> ground)
        was_on_ground = self.last_on_ground.get(player.car_id, True)

        # Flip detection: was in air, now on ground (simple heuristic)
        if not was_on_ground and player.on_ground:
            # Check if there was significant velocity change (flip boost)
            if player.car_id in self.last_player_velocity:
                vel_change = np.linalg.norm(
                    player.car_data.linear_velocity -
                    self.last_player_velocity[player.car_id]
                )
                if vel_change > 200:  # Threshold for flip detection
                    return 1.0
        return 0.0

    def _calculate_goal_side_positioning(self, player_pos: np.ndarray,
                                        ball_pos: np.ndarray,
                                        own_goal: np.ndarray) -> float:
        """Reward being between ball and own goal (defensive positioning)"""
        # Vector from goal to ball
        goal_to_ball = ball_pos - own_goal
        goal_to_ball_dist = np.linalg.norm(goal_to_ball)

        if goal_to_ball_dist < 1e-5:
            return 0.0

        goal_to_ball_norm = goal_to_ball / goal_to_ball_dist

        # Vector from goal to player
        goal_to_player = player_pos - own_goal

        # Project player position onto goal-ball line
        projection = np.dot(goal_to_player, goal_to_ball_norm)

        # Reward if player is between goal and ball
        if 0 < projection < goal_to_ball_dist:
            # Calculate how centered player is on the line
            closest_point = own_goal + goal_to_ball_norm * projection
            distance_from_line = np.linalg.norm(player_pos - closest_point)

            # Exponential decay based on distance from ideal line
            return np.exp(-distance_from_line / 800.0)
        return 0.0

    def _calculate_shadow_defense(self, player_pos: np.ndarray,
                                  ball_pos: np.ndarray,
                                  own_goal: np.ndarray,
                                  opponent_goal: np.ndarray) -> float:
        """Reward shadow defense positioning when ball is on opponent's side"""
        # Calculate field position of ball (-1 = own side, 1 = opponent side)
        field_length = np.linalg.norm(opponent_goal - own_goal)
        ball_to_own_goal = np.linalg.norm(ball_pos - own_goal)
        ball_field_position = (ball_to_own_goal / field_length) * 2 - 1

        # Only reward shadow defense when ball is on opponent's side
        if ball_field_position > 0.3:
            # Reward being closer to own goal than ball (ready to defend)
            player_to_own_goal = np.linalg.norm(player_pos - own_goal)

            if player_to_own_goal < ball_to_own_goal:
                # Distance from ideal shadow position (roughly midfield)
                ideal_distance = field_length * 0.35
                distance_diff = abs(player_to_own_goal - ideal_distance)

                # Exponential decay
                return np.exp(-distance_diff / 1500.0)
        return 0.0
