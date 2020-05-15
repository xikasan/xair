# coding: utf-8

from xair.models.lvaircraft import *

from gym.envs.registration import register

# =============================================================================
# lvaircraft pitch controls
# Rectangle command
register(
    id='LVAircraftPitch-v0',
    entry_point='xair.envs.lvaircraft_pitch:LVAircraftPitchV0'
)
# Filtered rectangle command
register(
    id='LVAircraftPitch-v1',
    entry_point='xair.envs.lvaircraft_pitch:LVAircraftPitchV1'
)
