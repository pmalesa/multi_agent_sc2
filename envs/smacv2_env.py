from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from absl import logging

logging.set_verbosity(logging.DEBUG)

def make_smacv2_env(distribution_config = None,
                    map_name = "10gen_terran",
                    debug = True,
                    conic_fov = False,
                    obs_own_pos = True,
                    use_unit_ranges = True,
                    min_attack_range = 2):
    
    if distribution_config is None:
        distribution_config = {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["medivac"],
                "weights": [1.0, 0.0, 0.0],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }

    env = StarCraftCapabilityEnvWrapper(
        capability_config = distribution_config,
        map_name = map_name,
        debug = debug,
        conic_fov = conic_fov,
        obs_own_pos = obs_own_pos,
        use_unit_ranges = use_unit_ranges,
        min_attack_range = min_attack_range
    )

    return env