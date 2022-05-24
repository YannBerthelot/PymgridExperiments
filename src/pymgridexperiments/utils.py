import os
import pickle
import random
import numpy as np
import pandas as pd
from copy import copy
from pymgrid import MicrogridGenerator as m_gen
from pymgrid.Environments.ScenarioEnvironment import CSPLAScenarioEnvironment
from pymgrid.Environments.MacroEnvironment import MacroEnvironment


def get_microgrid(id=1, export_price_factor=0.0):
    # Create 25 defaults microgrids
    env = m_gen.MicrogridGenerator(nb_microgrid=25)
    pymgrid25 = env.load("pymgrid25")
    microgrids_25 = pymgrid25.microgrids

    # Select the 1 microgrid
    mg = microgrids_25[1]

    # Modify export prices to not be 0
    mg._grid_price_export[0] = mg._grid_price_import[0] * export_price_factor
    mg._grid_status_ts[0] = pd.Series(np.ones(len(mg._grid_status_ts)))

    return mg


dir = os.path.dirname(__file__)
with open(os.path.join(dir, "data/fakeYears_archId1_15042022.pkl"), "rb") as f:
    fake_data = pickle.load(f)


def get_train_env(year=0, export_price_factor=0):
    starts = list(range(0, 6759, 2000))
    mg = get_microgrid(id=1, export_price_factor=export_price_factor)
    mg_train = copy(mg)
    return CSPLAScenarioEnvironment(
        starts,
        2000,
        {"microgrid": mg_train},
        fake_data["tsSample"][year][0][:, None],
        fake_data["tsSample"][year][1][:, None],
    )


def get_macro_environments(micropolicies):
    mg = get_microgrid(id=1, export_price_factor=0)
    mg_train = copy(mg)
    mg_test = copy(mg)

    mg_env_train = MacroEnvironment({"microgrid": mg_train}, micropolicies)
    mg_env_eval = MacroEnvironment({"microgrid": mg_test}, micropolicies)
    print(fake_data["tsSample"][0][1][:, None])
    return mg_env_train, mg_env_eval


def get_environments(pv_factor=1.0, action_design="original", export_price_factor=0):
    mg = get_microgrid(id=1, export_price_factor=export_price_factor)
    mg._data_set_to_use = "all"
    mg.dataset_to_use_default = "all"
    mg.TRAIN = False
    mg_train = copy(mg)
    mg_test = copy(mg)
    starts = list(range(0, 6759, 2000))
    starts = [0]
    mg_env_train = CSPLAScenarioEnvironment(
        starts,
        8760,
        {"microgrid": mg_train},
        # customPVTs=fake_data["tsSample"][0][1][:, None],
        # customLoadTs=fake_data["tsSample"][0][0][:, None],
        action_design=action_design,
    )

    mg_env_eval = CSPLAScenarioEnvironment(
        [0],
        8760,
        {"microgrid": mg_test},
        # customPVTs=fake_data["tsSample"][0][1][:, None],
        # customLoadTs=fake_data["tsSample"][0][0][:, None],
        action_design=action_design,
    )
    print(fake_data["tsSample"][0][1][:, None])
    return mg_env_train, mg_env_eval


def get_environments_for_cluster(
    cluster,
    pv_factor=1.0,
    starts_file="clusteringResultPymgrid25_configcfgN10k200_fakeYearsAssignmentNN.pkl",
    action_design="original",
    export_price_factor=0,
    seed=42,
):
    mg = get_microgrid(id=1, export_price_factor=export_price_factor)
    mg_train = copy(mg)
    mg_test = copy(mg)
    object = read_pickle(f"data/{starts_file}")[0]
    cluster_ids = object["clusterAssignments"][0]
    starts = object["pieceStarts"][0]
    max_cluster = max(cluster_ids)
    if cluster > max_cluster:
        raise ValueError(
            f"Cluster {cluster} does not exist, max cluster is {max_cluster}"
        )

    # Get the whole starts list for the current cluster
    starts_cluster = starts[cluster_ids == cluster]
    print(starts)
    # Split the starts list between train and test
    train_starts, test_starts = train_test_split(starts_cluster, seed=42)
    mg_env_train = CSPLAScenarioEnvironment(
        starts_cluster,
        1000,
        {"microgrid": mg_train},
        customPVTs=fake_data["tsSample"][0][1][:, None],
        customLoadTs=fake_data["tsSample"][0][0][:, None],
        action_design=action_design,
        pv_factor=pv_factor,
    )
    mg_env_eval = CSPLAScenarioEnvironment(
        [0],
        8760,
        {"microgrid": mg_test},
        # customPVTs=fake_data["tsSample"][0][1][:, None],
        # customLoadTs=fake_data["tsSample"][0][0][:, None],
        action_design=action_design,
    )

    return mg_env_train, mg_env_eval


def read_pickle(file):
    objects = []
    with (open(file, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
        return objects


def train_test_split(list_in, ratio=0.7, seed=None):
    if seed is not None:
        random.seed(seed)
    random.shuffle(list_in)
    split_idx = int(ratio * len(list_in))
    train = list_in[:split_idx]
    test = list_in[split_idx:]
    return train, test
