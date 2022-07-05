from deeprlyb.utils.config import read_config
from pymgrid_config import pymgrid_config
from utils import get_environments, log_episode, log_timestep
from pymgrid.Environments.rule_based import RuleBaseControl


def rule_based_test(env, logging):
    done, t = False, 0
    reward_sum = 0
    while not done:
        t += 1
        action = RuleBaseControl.rule_based_policy(policy="just_buy", mg=env.mg)
        next_state, reward, done, _ = env.step_RBC(action)
        reward_sum += reward
        log_timestep(reward, t, out=logging)
    log_episode(reward_sum, out=logging)


def rule_based_experiments(logging="tensorboard"):
    LOG_MODE = logging.lower()
    if LOG_MODE == "wandb":
        run = wandb.init(
            project="Pymgrid test RBC",
            entity="yann-berthelot",
            name="RuleBaseControl",
            reinit=True,
            config=pymgrid_config,
        )
    else:
        run = None
    mg_env_train, mg_env_eval = get_environments(
        pv_factor=pymgrid_config["pv_factor"],
        action_design=pymgrid_config["action_design"],
        export_price_factor=pymgrid_config["export_price_factor"],
    )

    rule_based_test(mg_env_eval, logging)

    if LOG_MODE == "wandb":
        run.finish()


if __name__ == "__main__":
    rule_based_experiments()
