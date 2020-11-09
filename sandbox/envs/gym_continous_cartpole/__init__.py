import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ContinuousCartPoleEnv-v0',
    entry_point='imitationingames.envs.gym_continous_cartpole.countinous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=500,
)
