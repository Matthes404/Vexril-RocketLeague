"""
Evaluation script for testing trained Rocket League bot
"""
import yaml
import numpy as np
import rlgym_sim
from pathlib import Path
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.obs_builders import DefaultObs

from src.agents.rl_bot import RLBot, RandomBot
from src.rewards.custom_reward import CustomReward
from src.state_setters.custom_state_setter import CustomStateSetter


def load_config(config_path: str = "configs/training_config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_bot(model_path: str, num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained bot

    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game (requires proper setup)
    """
    config = load_config()

    # Create environment
    print("Creating evaluation environment...")
    env = rlgym_sim.make(
        tick_skip=config['env']['tick_skip'],
        spawn_opponents=True,
        team_size=1,
        terminal_conditions=[
            TimeoutCondition(config['env']['timeout_seconds']),
            GoalScoredCondition()
        ],
        reward_fn=CustomReward(),
        obs_builder=DefaultObs(),
        state_setter=CustomStateSetter(),
        copy_gamestate_every_step=True
    )

    # Load bot
    print(f"Loading bot from {model_path}")
    bot = RLBot(model_path, deterministic=True)

    # Evaluation statistics
    total_rewards = []
    total_goals_scored = 0
    total_goals_conceded = 0
    episode_lengths = []

    print(f"\nStarting evaluation for {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        initial_blue_score = 0
        initial_orange_score = 0

        while not done:
            # Get action from bot
            action = bot.get_action(obs)

            # Step environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Track scores (assuming blue team)
            if episode_length == 1:
                initial_blue_score = info.get('blue_score', 0)
                initial_orange_score = info.get('orange_score', 0)

        # Calculate goals for this episode
        final_blue_score = info.get('blue_score', 0)
        final_orange_score = info.get('orange_score', 0)

        goals_scored = final_blue_score - initial_blue_score
        goals_conceded = final_orange_score - initial_orange_score

        total_goals_scored += goals_scored
        total_goals_conceded += goals_conceded
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Goals Scored={goals_scored}, "
              f"Goals Conceded={goals_conceded}, "
              f"Length={episode_length}")

    env.close()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"Total Goals Scored: {total_goals_scored}")
    print(f"Total Goals Conceded: {total_goals_conceded}")
    print(f"Goal Differential: {total_goals_scored - total_goals_conceded}")
    print("=" * 50)


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained Rocket League bot")
    parser.add_argument(
        "--model",
        type=str,
        default="models/rl_bot_final.zip",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the game (requires proper setup)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return

    evaluate_bot(args.model, args.episodes, args.render)


if __name__ == "__main__":
    main()
