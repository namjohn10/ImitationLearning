import os
from ray import tune
import copy

CONFIG = dict()

common_config = {
    "env": "MyEnv",
    "trainer_config": {},
    "env_config": {

    },
    "framework": "torch",
    "extra_python_environs_for_driver": {},
    "extra_python_environs_for_worker": {},
    "model": {
        "custom_model": "MyModel",
        "custom_model_config": {
            'value_function': None
        },
        "max_seq_len": 0    # Placeholder value needed for ray to register model
    },
    "evaluation_config": {},
}

CONFIG["ppo"] = copy.deepcopy(common_config)
CONFIG["ppo"]["trainer_config"]["algorithm"] = "PPO"
CONFIG["ppo"].update({
    # "horizon": inf,
    "horizon": 10000,
    "use_critic": True,
    "use_gae": True,
    "lambda": 0.99,
    "gamma": 0.99,
    "kl_coeff": 0.00,
    "shuffle_sequences": True,
    "num_sgd_iter": 10,
    "lr": 5e-5,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.000,
    "entropy_coeff_schedule": None,
    "clip_param": 0.2,
    "vf_clip_param": 100.0,
    "grad_clip": None,
    "kl_target": 0.01,
    "batch_mode":  "truncate_episodes",
    "observation_filter": "NoFilter",
    "normalize_actions": False,
    "clip_actions": True,

    # Device Configuration
    "create_env_on_driver": False,
    "num_cpus_for_driver": 0,
    "num_gpus": 1,
    "num_gpus_per_worker": 0.,
    "num_envs_per_worker": 1,
    "num_cpus_per_worker": 1,
})

# Muscle Configuration
CONFIG["ppo"]["trainer_config"]["muscle_lr"] = 5e-5
CONFIG["ppo"]["trainer_config"]["muscle_num_epochs"] = 10


# Large Set (For Cluster)
CONFIG["ppo_large"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_large"]["train_batch_size"] = 8192 * 8 * 4
CONFIG["ppo_large"]["sgd_minibatch_size"] = 4096
CONFIG["ppo_large"]["trainer_config"]["muscle_sgd_minibatch_size"] = 4096
CONFIG["ppo_large"]["trainer_config"]["marginal_sgd_minibatch_size"] = 4096
CONFIG["ppo_large"]["trainer_config"]["ref_sgd_minibatch_size"] = 4096

# Medium Set (For a node or a PC)
CONFIG["ppo_medium"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_medium"]["train_batch_size"] = 8192 * 4 * 2
CONFIG["ppo_medium"]["sgd_minibatch_size"] = 1024
CONFIG["ppo_medium"]["trainer_config"]["muscle_sgd_minibatch_size"] = 1024
CONFIG["ppo_medium"]["trainer_config"]["marginal_sgd_minibatch_size"] = 1024
CONFIG["ppo_medium"]["trainer_config"]["ref_sgd_minibatch_size"] = 1024

# ===============================Training Configuration For Various Devices=========================================

# Large Set
CONFIG["ppo_large_server"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_server"]["num_workers"] = 128 * 2

CONFIG["ppo_large_node"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_node"]["num_workers"] = 128

CONFIG["ppo_large_pc"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_pc"]["num_workers"] = 32

# Medium Set
CONFIG["ppo_medium_server"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_server"]["num_workers"] = 128 * 2

CONFIG["ppo_medium_node"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_node"]["num_workers"] = 128

CONFIG["ppo_medium_pc"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_pc"]["num_workers"] = 32