import subprocess
import os
import sys
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.abspath("/Users/fritz/SRLF"))
sys.path.append(os.path.abspath("/home/fritz/SRLF"))

import joblib
from io import BytesIO
import tensorflow as tf
import numpy as np


def launch_workers(worker_args, filename, wait=True):
    processes = []
    n_workers = worker_args['n_workers']
    for i in range(n_workers):
        cmd = 'python3 ' + filename
        for key, value in worker_args.items():
            cmd += ' --{arg_key}={arg_value}'.format(arg_key=key, arg_value=value)
        cmd += ' --worker_index={}'.format(i)
        processes.append(
            subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, preexec_fn=os.setsid))
    if wait:
        exit_codes = [p.wait() for p in processes]
        return exit_codes
    else:
        return processes


def agent_from_config(config):
    if config['trainer'] == 'ES':
        if config['continuous']:
            from algos.es_continuous import EvolutionStrategiesTrainer
            return EvolutionStrategiesTrainer
        else:
            from algos.es_discrete import EvolutionStrategiesTrainer
            return EvolutionStrategiesTrainer
    elif config['trainer'] == 'TRPO':
        config['critic'] = True
        if config['continuous']:
            from algos.trpo_continuous import TRPOContinuousTrainer
            return TRPOContinuousTrainer
        else:
            from algos.trpo_discrete import TRPODiscreteTrainer
            return TRPODiscreteTrainer
    elif config['trainer'] == 'DDPG':
        from algos.ddpg_distributed import DDPGTrainer
        return DDPGTrainer
    elif config['trainer'] == 'DDPG-S':
        from algos.ddpg_single import DDPGTrainer
        return DDPGTrainer
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
        from adapters.osim_adapter import OsimAdapter
        return OsimAdapter()


def dump_object(data):
    # converts whatever to string
    s = BytesIO()
    joblib.dump(data, s)

    return s.getvalue()


def load_object(string):
    # converts string to whatever was dumps'ed in it
    return joblib.load(BytesIO(string))


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


class SetFromFlat(object):
    def __init__(self, var_list, session):
        self.session = session
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list, session):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.session.run(self.op)


def discount(rewards, gamma, timestamps):
    dt = np.diff(timestamps.squeeze())
    x = rewards.squeeze()
    g = np.power(gamma, dt)
    y = np.zeros_like(x)
    for n in range(len(y)):
        y[n] = x[n] + np.sum(x[n + 1:] * np.cumprod(g[n:]))
    return y


def linesearch(f, x, fullstep, max_kl):
    max_backtracks = 10
    loss, _ = f(x)
    for stepfrac in .5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        new_loss, kl = f(xnew)
        actual_improve = new_loss - loss
        if kl <= max_kl and actual_improve < 0:
            x = xnew
            loss = new_loss
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (p.dot(z) + 1e-18)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + 1e-18)
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x
