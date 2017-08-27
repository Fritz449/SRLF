import subprocess
import os
import sys
sys.path.append(os.path.realpath(".."))

import joblib
from io import BytesIO
sys.path.append(os.path.abspath("/Users/fritz/SRLF"))


def launch_rollout_es_workers(worker_args, filename):
    processes = []
    n_workers = worker_args['n_workers']
    for i in range(n_workers):
        cmd = 'python3 ' + filename
        for key, value in worker_args.items():
            cmd += ' --{arg_key}={arg_value}'.format(arg_key=key, arg_value=value)
        cmd += ' --worker_index={}'.format(i)
        processes.append(
            subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, preexec_fn=os.setsid))
    exit_codes = [p.wait() for p in processes]
    return exit_codes


def agent_from_config(config):
    if config['trainer'] == 'ES':
        if config['continuous']:
            from algos.es_continuous import EvolutionStrategiesTrainer
            return EvolutionStrategiesTrainer
        else:
            from algos.es_discrete import EvolutionStrategiesTrainer
            return EvolutionStrategiesTrainer
    else:
        raise Exception


def env_from_config(config):
    if config['env_type'] == 'gym':
        if config['continuous']:
            from adapters.gym_continuous import GymAdapterContinuous
            return GymAdapterContinuous(config['env_name'])
        else:
            from adapters.gym_discrete import GymAdapterDiscrete
            return GymAdapterDiscrete(config['env_name'])
    else:
        raise Exception


def dump_object(data):
    # converts whatever to string
    s = BytesIO()
    joblib.dump(data, s)

    return s.getvalue()


def load_object(string):
    # converts string to whatever was dumps'ed in it
    return joblib.load(BytesIO(string))
