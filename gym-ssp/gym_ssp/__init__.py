from gym.envs.registration import register

register(
    id='ssp-v0',
    entry_point='gym_ssp.envs:SSP'
)