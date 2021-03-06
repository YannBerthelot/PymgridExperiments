import os
import wandb
from deeprlyb.agents.A2C import A2C
from deeprlyb.utils.config import read_config
from pymgrid_config import pymgrid_config
from utils import get_environments, get_macro_environments


def pymgrid_macro():
    config = read_config()
    config["GLOBAL"]["name"] = "pymgrid"
    config["GLOBAL"]["environment"] = "pymgrid"
    config_global = {**config, **pymgrid_config}
    for n_step in [1, 5, 24, 48, 24 * 7]:
        config_global["GLOBAL"]["n_steps"] = str(n_step)
        for experiment in range(1, config["GLOBAL"].getint("n_experiments") + 1):
            if config["GLOBAL"]["logging"] == "wandb":
                run = wandb.init(
                    project="Pymgrid macro-agent boosted",
                    entity="yann-berthelot",
                    name=f'{config["GLOBAL"]["name"]} {experiment}/{config["GLOBAL"]["n_experiments"]}',
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
            micro_policies = []
            # for save in os.listdir("models"):
            #     if (save.split("_")[2] == "1") and (save.split("_")[0] == "cluster"):
            #         micro_agent = A2C(
            #             mg_env_train,
            #             config=config,
            #         )
            #         micro_agent.load(save.split(".")[0])
            #         micro_policies.append(micro_agent)

            mg_env_train, mg_env_eval = get_macro_environments(micro_policies)
            macro_agent = A2C(
                mg_env_train,
                config=config,
                comment=f"macro_agent_{experiment}",
                run=run,
            )
            macro_agent.run = run
            macro_agent.train_TD0(
                mg_env_train, config["GLOBAL"].getfloat("nb_timesteps_train")
            )

            # Load best agent from training
            macro_agent.env = mg_env_eval
            macro_agent.load(f"macro_agent_{experiment}_best")

            # Evaluate and render the policy
            macro_agent.test(
                mg_env_eval,
                nb_episodes=config["GLOBAL"].getint("nb_episodes_test"),
                render=False,
                # scaler_file=f"data/baseline_pymgrid_{experiment}_obs_scaler.pkl",
            )
            if config["GLOBAL"]["logging"] == "wandb":
                run.finish()


if __name__ == "__main__":
    pymgrid_macro()
