import os
import pdb
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
            # mg_env_train_plus, mg_env_eval_plus = get_environments(
            #     pv_factor=pymgrid_config["pv_factor"],
            #     action_design=pymgrid_config["action_design"],
            #     export_price_factor=pymgrid_config["export_price_factor"],
            # )
            # mg_env_train_minus, mg_env_eval_minus = get_environments(
            #     pv_factor=pymgrid_config["pv_factor"],
            #     action_design=pymgrid_config["action_design"],
            #     export_price_factor=-pymgrid_config["export_price_factor"],
            # )
            micro_policies = []
            # for save in os.listdir("models"):
            #     if (save.split("_")[2] == "1") and (save.split("_")[0] == "cluster"):
            #         micro_agent = A2C(
            #             mg_env_train,
            #             config=config,
            #         )
            #         micro_agent.load(save.split(".")[0])
            #         micro_policies.append(micro_agent)

            mg_env_train_plus, mg_env_eval_plus = get_macro_environments(
                micro_policies,
                export_price_factor=pymgrid_config["export_price_factor"],
                pv_factor=pymgrid_config["pv_factor"],
            )
            mg_env_train_min, mg_env_eval_plus_min = get_macro_environments(
                micro_policies,
                export_price_factor=-pymgrid_config["export_price_factor"],
                pv_factor=pymgrid_config["pv_factor"],
            )
            macro_agent = A2C(
                mg_env_train_plus,
                config=config,
                comment=f"macro_agent_{experiment}",
                run=run,
            )
            macro_agent.run = run
            macro_agent.pre_train(
                mg_env_train_plus,
                config["GLOBAL"].getfloat("learning_start"),
                scaling=config["GLOBAL"].getboolean("scaling"),
            )

            mg_env_train_plus._max_episode_steps = 8738
            mg_env_train_min._max_episode_steps = 8738
            N_ITER = (
                int(config["GLOBAL"].getfloat("nb_timesteps_train"))
                // mg_env_train_plus._max_episode_steps
            )
            nb_timesteps = (mg_env_train_plus._max_episode_steps - 1) * 2
            for i in range(1, N_ITER - 2, 2):
                macro_agent.train_TD0(
                    mg_env_train_plus,
                    nb_timesteps,
                )
                macro_agent.train_TD0(
                    mg_env_train_min,
                    nb_timesteps,
                )

            # Load best agent from training
            macro_agent.env = mg_env_eval_plus
            macro_agent.load(f"macro_agent_{experiment}_best")

            # Evaluate and render the policy
            macro_agent.test(
                mg_env_eval_plus,
                nb_episodes=config["GLOBAL"].getint("nb_episodes_test"),
                render=False,
                # scaler_file=f"data/baseline_pymgrid_{experiment}_obs_scaler.pkl",
            )
            if config["GLOBAL"]["logging"] == "wandb":
                run.finish()


if __name__ == "__main__":
    pymgrid_macro()
