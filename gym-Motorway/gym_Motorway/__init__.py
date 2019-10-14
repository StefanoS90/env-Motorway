# -*- coding: utf-8 -*-

from gym.envs.registration import register
from gym_Motorway.envs.Motorway_env import MotorwayEnv

register(
    id='Motorway-v0',
    entry_point='gym_Motorway.envs:MotorwayEnv',
)
