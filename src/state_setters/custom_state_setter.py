"""
Custom state setter for Rocket League bot training
Sets up initial game states for varied training scenarios
"""
import numpy as np
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.common_values import CEILING_Z, BACK_WALL_Y, SIDE_WALL_X, BALL_RADIUS


class CustomStateSetter(StateSetter):
    """
    A custom state setter that creates varied training scenarios:
    - Random ball positions (on ground and in air)
    - Random player positions around the field
    - Random ball and player velocities

    This helps the bot learn in diverse situations.
    """

    def __init__(self, random_boost: bool = True, random_ball_height: bool = True):
        """
        Args:
            random_boost: If True, randomize player boost amounts
            random_ball_height: If True, spawn ball at various heights
        """
        super().__init__()
        self.random_boost = random_boost
        self.random_ball_height = random_ball_height
        self.default_setter = DefaultState()

    def reset(self, state_wrapper):
        """
        Reset the game state with custom initialization

        Args:
            state_wrapper: The state wrapper object to modify
        """
        # 20% chance to use default kickoff state
        if np.random.random() < 0.2:
            self.default_setter.reset(state_wrapper)
            return

        # Set ball position
        ball_x = np.random.uniform(-SIDE_WALL_X * 0.7, SIDE_WALL_X * 0.7)
        ball_y = np.random.uniform(-BACK_WALL_Y * 0.7, BACK_WALL_Y * 0.7)

        if self.random_ball_height:
            # Spawn ball at various heights (weighted towards ground)
            height_choice = np.random.random()
            if height_choice < 0.5:
                # Ground level
                ball_z = BALL_RADIUS
            elif height_choice < 0.8:
                # Low-medium height
                ball_z = np.random.uniform(BALL_RADIUS, 500)
            else:
                # High in the air
                ball_z = np.random.uniform(500, CEILING_Z * 0.8)
        else:
            ball_z = BALL_RADIUS

        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)

        # Set ball velocity (sometimes moving, sometimes stationary)
        if np.random.random() < 0.7:
            ball_vel_x = np.random.uniform(-1000, 1000)
            ball_vel_y = np.random.uniform(-1000, 1000)
            ball_vel_z = np.random.uniform(-500, 500)
            state_wrapper.ball.set_lin_vel(ball_vel_x, ball_vel_y, ball_vel_z)
        else:
            state_wrapper.ball.set_lin_vel(0, 0, 0)

        # Set ball angular velocity
        state_wrapper.ball.set_ang_vel(
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2)
        )

        # Set player states
        for car in state_wrapper.cars:
            # Random position on the field
            car_x = np.random.uniform(-SIDE_WALL_X * 0.8, SIDE_WALL_X * 0.8)
            car_y = np.random.uniform(-BACK_WALL_Y * 0.8, BACK_WALL_Y * 0.8)
            car_z = 17  # Standard car height on ground

            car.set_pos(car_x, car_y, car_z)

            # Random orientation (yaw)
            yaw = np.random.uniform(-np.pi, np.pi)
            pitch = 0
            roll = 0
            car.set_rot(pitch, yaw, roll)

            # Random velocity (sometimes stationary)
            if np.random.random() < 0.6:
                vel_magnitude = np.random.uniform(0, 1500)
                vel_angle = np.random.uniform(-np.pi, np.pi)
                car.set_lin_vel(
                    vel_magnitude * np.cos(vel_angle),
                    vel_magnitude * np.sin(vel_angle),
                    0
                )
            else:
                car.set_lin_vel(0, 0, 0)

            # Random boost amount
            if self.random_boost:
                car.boost = np.random.uniform(0, 100)
            else:
                car.boost = 33  # Standard starting boost

            # Reset other car states
            car.set_ang_vel(0, 0, 0)
            car.jumped = False
            car.double_jumped = False
            car.on_ground = True
