import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK


class CustomReward(RewardFunction):
    """
    Improved Rocket League reward function for 1v1 PPO training.

    Optimized for FIRST training run with focus on:
    1. Strong shaping signals to guide early exploration
    2. Boost management fundamentals
    3. Progressive ball interaction rewards
    4. Touch quality scaling (harder hits = more reward)
    5. Basic positional awareness

    Reward Philosophy:
    - Shaping rewards are continuous and guide behavior
    - Sparse rewards are milestones that reinforce good play
    - All rewards are scaled to work well with PPO and VecNormalize
    """

    def __init__(self):
        super().__init__()

        # ==============================================
        # SPARSE EVENT REWARDS (milestone achievements)
        # ==============================================

        # Ball interaction (scaled by touch quality)
        self.touch_base_reward = 8.0       # Base reward for any touch
        self.touch_velocity_bonus = 15.0   # Bonus scaled by hit strength
        self.shot_reward = 25.0            # Ball moving toward goal after touch
        self.aerial_touch_bonus = 10.0     # Bonus for touching ball in air

        # Goal events
        self.goal_reward = 100.0           # Scoring a goal
        self.concede_penalty = -100.0      # Opponent scores

        # Defensive play
        self.save_reward = 35.0            # Clearing ball moving toward own goal

        # Boost management
        self.big_boost_reward = 2.0        # Collecting 100 boost pad
        self.small_boost_reward = 0.5      # Collecting small boost pad

        # ==============================================
        # SHAPING REWARDS (continuous guidance signals)
        # ==============================================

        # Ball-focused shaping (highest priority for first training)
        self.ball_vel_to_goal_weight = 0.20    # Ball moving toward opponent goal
        self.car_vel_to_ball_weight = 0.15     # Car velocity toward ball
        self.car_face_ball_weight = 0.10       # Car orientation toward ball

        # Distance shaping (exponential - stronger when close)
        self.distance_to_ball_weight = 0.12   # Reward for being near ball

        # Speed/momentum shaping
        self.car_speed_weight = 0.05          # Reward for maintaining speed

        # Positioning shaping
        self.ball_height_weight = 0.03        # Small reward when ball is airborne
        self.offensive_position_weight = 0.04  # Reward for being on offensive half

        # ==============================================
        # PENALTY WEIGHTS (discourage bad behavior)
        # ==============================================
        self.ball_vel_to_own_goal_penalty = 0.08  # Penalize ball moving to own goal
        self.idle_penalty = 0.02                   # Small penalty for low speed

        # ==============================================
        # TRACKING VARIABLES
        # ==============================================
        self.last_ball_vel = None
        self.last_touched = {}
        self.last_ball_pos = None
        self.last_boost_amount = {}
        self.last_car_speed = {}

    def reset(self, initial_state: GameState):
        """Reset tracking variables at episode start."""
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_ball_pos = initial_state.ball.position.copy()
        self.last_touched = {p.car_id: False for p in initial_state.players}
        self.last_boost_amount = {p.car_id: p.boost_amount for p in initial_state.players}
        self.last_car_speed = {p.car_id: 0.0 for p in initial_state.players}

    def get_reward(self, player: PlayerData, state: GameState, prev_action):
        """Calculate per-step reward for training."""
        r = 0.0

        ball = state.ball
        car = player.car_data

        ball_pos = ball.position
        ball_vel = ball.linear_velocity
        car_pos = car.position
        car_vel = car.linear_velocity
        car_forward = car.forward()

        # Precompute commonly used values
        car_speed = np.linalg.norm(car_vel)
        ball_speed = np.linalg.norm(ball_vel)

        # Goal positions
        opponent_goal = (
            np.array(ORANGE_GOAL_BACK)
            if player.team_num == BLUE_TEAM
            else np.array(BLUE_GOAL_BACK)
        )
        own_goal = (
            np.array(BLUE_GOAL_BACK)
            if player.team_num == BLUE_TEAM
            else np.array(ORANGE_GOAL_BACK)
        )

        # Direction vectors
        dir_to_ball = ball_pos - car_pos
        dist_to_ball = np.linalg.norm(dir_to_ball)
        dir_to_ball_norm = dir_to_ball / (dist_to_ball + 1e-6)

        dir_ball_to_opponent_goal = opponent_goal - ball_pos
        dist_ball_to_opponent_goal = np.linalg.norm(dir_ball_to_opponent_goal)
        dir_ball_to_opponent_goal_norm = dir_ball_to_opponent_goal / (dist_ball_to_opponent_goal + 1e-6)

        dir_ball_to_own_goal = own_goal - ball_pos
        dist_ball_to_own_goal = np.linalg.norm(dir_ball_to_own_goal)
        dir_ball_to_own_goal_norm = dir_ball_to_own_goal / (dist_ball_to_own_goal + 1e-6)

        # ==========================================
        # 1. BALL TOUCH REWARDS (Sparse + Quality)
        # ==========================================
        touch_occurred = player.ball_touched and not self.last_touched.get(player.car_id, False)

        if touch_occurred:
            # Base touch reward
            r += self.touch_base_reward

            # Touch velocity bonus (scaled by how hard the hit was)
            # Compare current ball speed to previous - stronger hits = more reward
            if self.last_ball_vel is not None:
                velocity_change = ball_speed - np.linalg.norm(self.last_ball_vel)
                # Normalize by max ball speed (~6000 uu/s)
                velocity_bonus = np.clip(velocity_change / 3000.0, 0, 1) * self.touch_velocity_bonus
                r += velocity_bonus

            # Aerial touch bonus (ball height > 300 units)
            if ball_pos[2] > 300:
                # Scale bonus by height (more height = more reward)
                height_factor = np.clip((ball_pos[2] - 300) / 1000.0, 0, 1)
                r += self.aerial_touch_bonus * (0.5 + 0.5 * height_factor)

            # Shot detection: ball moving toward goal after touch
            ball_vel_norm = ball_vel / (ball_speed + 1e-6)
            shot_alignment = np.dot(ball_vel_norm, dir_ball_to_opponent_goal_norm)

            # Require good alignment (>0.5) and decent ball speed (>500 uu/s)
            if shot_alignment > 0.5 and ball_speed > 500:
                # Scale shot reward by alignment quality
                shot_quality = (shot_alignment - 0.5) * 2  # 0 to 1
                r += self.shot_reward * (0.5 + 0.5 * shot_quality)

        self.last_touched[player.car_id] = player.ball_touched

        # ==========================================
        # 2. BALL VELOCITY TOWARD OPPONENT GOAL
        # ==========================================
        vel_to_opponent_goal = np.dot(ball_vel, dir_ball_to_opponent_goal_norm)
        vel_to_opponent_goal_norm = np.clip(vel_to_opponent_goal / 4000.0, -1, 1)
        r += vel_to_opponent_goal_norm * self.ball_vel_to_goal_weight

        # ==========================================
        # 3. BALL VELOCITY TOWARD OWN GOAL (Penalty)
        # ==========================================
        vel_to_own_goal = np.dot(ball_vel, dir_ball_to_own_goal_norm)
        if vel_to_own_goal > 0:
            vel_to_own_goal_norm = np.clip(vel_to_own_goal / 4000.0, 0, 1)
            r -= vel_to_own_goal_norm * self.ball_vel_to_own_goal_penalty

        # ==========================================
        # 4. CAR VELOCITY TOWARD BALL
        # ==========================================
        vel_to_ball = np.dot(car_vel, dir_to_ball_norm)
        vel_to_ball_norm = np.clip(vel_to_ball / 2300.0, -1, 1)
        r += vel_to_ball_norm * self.car_vel_to_ball_weight

        # ==========================================
        # 5. CAR FACING BALL
        # ==========================================
        face_ball_alignment = np.dot(car_forward, dir_to_ball_norm)
        # Normalize from [-1, 1] to [0, 1]
        face_ball_reward = (face_ball_alignment + 1) / 2
        r += face_ball_reward * self.car_face_ball_weight

        # ==========================================
        # 6. DISTANCE TO BALL (Exponential scaling)
        # ==========================================
        # Use exponential decay - reward is highest when very close
        # exp(-dist/2000) gives ~0.6 at 1000 units, ~0.37 at 2000, ~0.05 at 6000
        dist_reward = np.exp(-dist_to_ball / 2500.0)
        r += dist_reward * self.distance_to_ball_weight

        # ==========================================
        # 7. CAR SPEED REWARD
        # ==========================================
        # Encourage maintaining speed (supersonic is ~2200 uu/s)
        speed_norm = np.clip(car_speed / 2300.0, 0, 1)
        r += speed_norm * self.car_speed_weight

        # Small penalty for being nearly stationary
        if car_speed < 100:
            r -= self.idle_penalty

        # ==========================================
        # 8. BOOST COLLECTION REWARDS
        # ==========================================
        if player.car_id in self.last_boost_amount:
            boost_gained = player.boost_amount - self.last_boost_amount[player.car_id]
            if boost_gained > 0:
                # Big boost pads give ~34 boost, small give ~12
                if boost_gained > 30:
                    r += self.big_boost_reward
                elif boost_gained > 5:
                    r += self.small_boost_reward

        self.last_boost_amount[player.car_id] = player.boost_amount

        # ==========================================
        # 9. BALL HEIGHT BONUS (Encourage aerials)
        # ==========================================
        # Small reward when ball is in air - encourages learning to hit aerials
        if ball_pos[2] > 200:
            height_bonus = np.clip((ball_pos[2] - 200) / 1500.0, 0, 1)
            r += height_bonus * self.ball_height_weight

        # ==========================================
        # 10. OFFENSIVE POSITIONING
        # ==========================================
        # Reward being on offensive half of field
        # Blue team attacks toward positive Y, Orange toward negative Y
        if player.team_num == BLUE_TEAM:
            offensive_pos = car_pos[1] / 5120.0  # Normalize by half-field length
        else:
            offensive_pos = -car_pos[1] / 5120.0

        offensive_pos_clipped = np.clip(offensive_pos, -1, 1)
        r += offensive_pos_clipped * self.offensive_position_weight

        # ==========================================
        # 11. SAVE DETECTION (Defensive Play)
        # ==========================================
        if self.last_ball_pos is not None and touch_occurred:
            # Check if ball was threatening own goal
            last_dir_to_own = own_goal - self.last_ball_pos
            last_dist_to_own = np.linalg.norm(last_dir_to_own)
            last_dir_to_own_norm = last_dir_to_own / (last_dist_to_own + 1e-6)

            last_vel_to_own_goal = np.dot(self.last_ball_vel, last_dir_to_own_norm)

            # Ball was moving toward own goal and now isn't
            current_vel_to_own_goal = np.dot(ball_vel, dir_ball_to_own_goal_norm)

            # Trigger save reward if:
            # 1. Ball was moving toward own goal at decent speed (>800 uu/s)
            # 2. Ball is now moving away or slower
            # 3. Ball was reasonably close to goal (within 4000 units)
            if (last_vel_to_own_goal > 800 and
                current_vel_to_own_goal < last_vel_to_own_goal * 0.3 and
                last_dist_to_own < 4000):
                r += self.save_reward

        # ==========================================
        # UPDATE TRACKING VARIABLES
        # ==========================================
        self.last_ball_vel = ball_vel.copy()
        self.last_ball_pos = ball_pos.copy()
        self.last_car_speed[player.car_id] = car_speed

        return r

    def get_final_reward(self, player: PlayerData, state: GameState, prev_action):
        """Calculate terminal reward when episode ends (goal scored)."""
        r = 0.0

        if player.team_num == BLUE_TEAM:
            if state.blue_score > 0:
                r += self.goal_reward
            elif state.orange_score > 0:
                r += self.concede_penalty
        else:
            if state.orange_score > 0:
                r += self.goal_reward
            elif state.blue_score > 0:
                r += self.concede_penalty

        return r
