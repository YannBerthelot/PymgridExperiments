import wandb
from deeprlyb.agents.A2C import A2C
from deeprlyb.utils.config import read_config
from pymgrid_config import pymgrid_config
from utils import get_environments


def pymgrid_experiment():
    config = read_config()
    config["GLOBAL"]["name"] = "pymgrid"
    config["GLOBAL"]["environment"] = "pymgrid"
    config_global = {**config, **pymgrid_config}
    for n_step in [1]:
        config_global["GLOBAL"]["n_steps"] = str(n_step)
        for experiment in range(1, config["GLOBAL"].getint("n_experiments") + 1):
            if config["GLOBAL"]["logging"] == "wandb":
                run = wandb.init(
                    project="Pymgrid test RBC",
                    entity="yann-berthelot",
                    name=f'{config["GLOBAL"]["name"]} {experiment}/{config["GLOBAL"].getint("N_EXPERIMENTS")}',
                    reinit=True,
                    config=config_global,
                )
            else:
                run = None
            mg_env_train, mg_env_eval = get_environments(
                pv_factor=pymgrid_config["pv_factor"],
                action_design=pymgrid_config["action_design"],
                export_price_factor=pymgrid_config["export_price_factor"],
            )
            agent = A2C(
                mg_env_train,
                config,
                comment=f"baseline_pymgrid_{experiment}",
                run=run,
            )
            agent.run = run
            agent.train_TD0(
                mg_env_train, config["GLOBAL"].getfloat("NB_TIMESTEPS_TRAIN")
            )
            agent.env = mg_env_eval
            agent.load(f"baseline_pymgrid_{experiment}_best")
            agent.test(
                mg_env_eval,
                nb_episodes=config["GLOBAL"].getint("NB_EPISODES_TEST"),
                render=False,
            )
            if config["GLOBAL"]["logging"] == "wandb":
                run.finish()


if __name__ == "__main__":
    pymgrid_experiment()
