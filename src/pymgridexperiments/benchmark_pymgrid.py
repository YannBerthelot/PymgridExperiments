import pymgrid
from pymgrid_config import pymgrid_config
from pymgrid_utils import get_environments
from pymgrid.algos.Control import Benchmarks

mg_env_train, mg_env_eval = get_environments(
    pv_factor=pymgrid_config["pv_factor"],
    export_price_factor=pymgrid_config["export_price_factor"],
)

benchmark = Benchmarks(mg_env_eval.mg)
benchmark.run_rule_based_benchmark()
benchmark.describe_benchmarks()
# benchmark.run_mpc_benchmark(verbose=True)
# benchmark.describe_benchmarks()
