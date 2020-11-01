#from gym.envs.registration import register
#
#
#register(
#    id="objcollection-v0",
#    entry_point="gym_objcollection.envs:ObjCollectionEnv",
#    timestep_limit=50)

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='objcollection-v0',
    entry_point='gym_objcollection.envs:ObjCollectionEnv',
)
