PPO_default = {
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
}

PPO_optimized = {
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 25,
    "gamma": 0.9908980966893566,
    "gae_lambda": 0.98,
    "ent_coef": 0.01,
}

PPO_optimized_a66 = {
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "ent_coef": 0.01,
}

PPO_optimized_gab = {
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 8,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "ent_coef": 0.01,
}

DQN_default = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "gamma": 0.99,
    "exploration_final_eps": 0.05
}

DQN_optimized = {
    "learning_rate": 0.00073739,
    "batch_size": 64,
    "gamma": 0.9906,
    "exploration_final_eps": 0.09467
}

A2C_default = {
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.0
}

A2C_optimized = {
    "learning_rate": 0.000566,
    "n_steps": 10,
    "gamma": 0.991,
    "gae_lambda": 0.953,
    "ent_coef": 0.01
}

HYPERPARAMS = {
    "PPO_default": PPO_default,
    "PPO_optimized": PPO_optimized,
    "PPO_optimized_a66": PPO_optimized_a66,
    "PPO_optimized_gab": PPO_optimized_gab,
    "DQN_default": DQN_default,
    "DQN_optimized": DQN_optimized,
    "A2C_default": A2C_default,
    "A2C_optimized": A2C_optimized,
}
