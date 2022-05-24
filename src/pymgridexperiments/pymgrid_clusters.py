import wandb
from A2C import A2C
from config import config
from pymgrid_utils import get_environments_for_cluster
from pymgrid_config import pymgrid_config

config["name"] = "pymgrid"
baseline_model = "baseline_pymgrid_1_best"


N_CLUSTERS = 10
for cluster in range(N_CLUSTERS):
    mg_env_train, mg_env_eval = get_environments_for_cluster(
        cluster,
        pv_factor=pymgrid_config["pv_factor"],
        action_design=pymgrid_config["action_design"],
    )
    config["CLUSTER"] = cluster
    config_global = {**config, **pymgrid_config}

    for experiment in range(1, config["N_EXPERIMENTS"] + 1):
        # if experiment == 1:
        #     config["BASELINE"] = True
        #     run = wandb.init(
        #         project="Pymgrid fake data clusters 2",
        #         entity="yann-berthelot",
        #         name=f"cluster_{cluster}_baseline",
        #         reinit=True,
        #         config=config_global,
        #     )
        #     agent = A2C(
        #         mg_env_eval,
        #         config=config,
        #         comment=f"cluster_{cluster}_baseline",
        #         run=run,
        #     )
        #     agent.load(f"{baseline_model}")
        #     agent.comment = f"baseline_cluster_{cluster}"
        #     agent.test(
        #         mg_env_eval,
        #         nb_episodes=config["NB_EPISODES_TEST"],
        #         render=False,
        #         scaler_file="data/baseline_pymgrid_1_obs_scaler.pkl",
        #     )
        #     if config["logging"] == "wandb":
        #         run.finish()
        config["BASELINE"] = False
        run = wandb.init(
            project="Pymgrid fake data clusters",
            entity="yann-berthelot",
            name=f'Cluster {cluster} {experiment}/{config["N_EXPERIMENTS"]}',
            reinit=True,
            config=config,
        )

        agent = A2C(
            mg_env_train,
            config=config,
            comment=f"cluster_{cluster}_{experiment}",
            run=run,
        )

        # Train the agent
        agent.train(mg_env_train, config["NB_TIMESTEPS_TRAIN"])

        # Load best agent from training
        agent.env = mg_env_eval
        agent.load(f"cluster_{cluster}_{experiment}_best")

        # Evaluate and render the policy
        agent.test(mg_env_eval, nb_episodes=config["NB_EPISODES_TEST"], render=False)

        if config["logging"] == "wandb":
            run.finish()
