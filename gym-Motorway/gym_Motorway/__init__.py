# -*- coding: utf-8 -*-

from gym.envs.registration import register

register(
    id='Motorway-v0',
    entry_point='gym_Motorway.envs:MotorwayEnv',
)
