from gym.envs.registration import register

register(
    id='deep-sea-treasure-v0',
    entry_point='deep_sea_treasure.envs:DeepSeaTreasureEnv',
)
register(
    id='bountiful-sea-treasure-v0',
    entry_point='deep_sea_treasure.envs:BountyfulSeaTreasureEnv',
)