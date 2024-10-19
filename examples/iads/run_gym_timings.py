"""Runner for the AI Gym run of IADS."""
import copy
from typing import List

from gym_ret.envs.ret_env import RetEnv

from iads.model import IADS


def run_one_step():
    """Run one step of the model."""
    env = RetEnv(IADS)
    obs = env.reset()

    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    print(rewards)
    return rewards

def run_n_steps(n: int) -> List[float]:
    """Run n steps of the model."""
    env = RetEnv(IADS)
    obs = env.reset()

    rewards = []
    for  _ in range(n):
        # experiment with making a deep copy of the environment
        # env = copy.deepcopy(env)
        action = env.action_space.sample()

        try:
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        except Exception as e:
            print(f"Error: {e}")
            break
    return rewards


if __name__ == "__main__":
    # time running n steps
    import time
    start = time.time()
    rewards = run_n_steps(100)
    end = time.time()
    print(rewards)
    print(len(rewards))
    print(type(rewards))
    print(f"Time taken: {end - start}")
    # average time per step
    print(f"Average time per step: {(end - start) / len(rewards)}")

