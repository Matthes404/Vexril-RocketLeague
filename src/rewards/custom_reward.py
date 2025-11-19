import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK


class CustomReward(RewardFunction):
    """
    Clean, high-signal Rocket League reward function for 1v1 PPO training.
    Designed for stability, simplicity, and real gameplay emergence.
    """

    def __init__(self):
        super().__init__()

        # Sparse events
        self.touch_reward = 8.0
        self.shot_reward = 25.0
        self.goal_reward = 120.0
        self.concede_penalty = -120.0

        # Shaping (small)
        self.ball_vel_to_goal_weight = 0.1
        self.car_vel_to_ball_weight = 0.08
        self.car_face_ball_weight = 0.05

        # Tracking
        self.last_ball_vel = None
        self.last_touched = {}

    # -------------------------------------------------------

    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_touched = {p.car_id: False for p in initial_state.players}

    # -------------------------------------------------------

    def get_reward(self, player: PlayerData, state: GameState, prev_action):
        r = 0.0

        ball = state.ball
        car = player.car_data

        ball_pos = ball.position
        ball_vel = ball.linear_velocity
        car_pos = car.position
        car_vel = car.linear_velocity
        car_forward = car.forward()

        # Opponent goal
        opponent_goal = (
            np.array(ORANGE_GOAL_BACK)
            if player.team_num == BLUE_TEAM
            else np.array(BLUE_GOAL_BACK)
        )

        # ------------------------------
        # 1. Sparse: Ball Touch Reward
        # ------------------------------
        if player.ball_touched and not self.last_touched[player.car_id]:
            r += self.touch_reward

            # Shot detection: ball moves toward goal WITH significant velocity
            dir_to_goal = opponent_goal - ball_pos
            dir_norm = dir_to_goal / (np.linalg.norm(dir_to_goal) + 1e-6)
            alignment = np.dot(ball_vel / (np.linalg.norm(ball_vel) + 1e-6), dir_norm)

            if alignment > 0.6:
                r += self.shot_reward

            self.last_touched[player.car_id] = True
        else:
            self.last_touched[player.car_id] = False

        # ------------------------------
        # 2. Shaping: Ball vel → goal
        # ------------------------------
        vel_to_goal = np.dot(ball_vel, (opponent_goal - ball_pos)) / (
            np.linalg.norm(opponent_goal - ball_pos) + 1e-6
        )

        # Normalize: ball max speed ≈ 6000
        vel_to_goal_norm = np.clip(vel_to_goal / 6000.0, -1, 1)
        r += vel_to_goal_norm * self.ball_vel_to_goal_weight

        # ------------------------------
        # 3. Shaping: Car vel → ball
        # ------------------------------
        dir_to_ball = (ball_pos - car_pos)
        dir_to_ball_norm = dir_to_ball / (np.linalg.norm(dir_to_ball) + 1e-6)

        vel_to_ball = np.dot(car_vel, dir_to_ball_norm)
        vel_to_ball_norm = np.clip(vel_to_ball / 2300.0, -1, 1)
        r += vel_to_ball_norm * self.car_vel_to_ball_weight

        # ------------------------------
        # 4. Shaping: Car facing ball
        # ------------------------------
        face_ball_alignment = np.dot(car_forward, dir_to_ball_norm)
        r += ((face_ball_alignment + 1) / 2) * self.car_face_ball_weight

        # ------------------------------
        # Update trackers
        # ------------------------------
        self.last_ball_vel = ball_vel.copy()

        return r

    # -------------------------------------------------------

    def get_final_reward(self, player: PlayerData, state: GameState, prev_action):
        r = 0.0

        # Team scored?
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
