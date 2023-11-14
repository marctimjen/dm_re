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

HYPERPARAMS = {
    "PPO_default": PPO_default,
    "PPO_optimized": PPO_optimized,
    "PPO_optimized_a66": PPO_optimized_a66,
    "PPO_optimized_gab": PPO_optimized_gab
}
