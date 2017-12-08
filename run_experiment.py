import json
import os
import sys
import helpers.utils as hlp
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.app.flags.DEFINE_string("config", 'a3c_c_lland.json', "What config the agent should use")
tf.app.flags.DEFINE_integer("id_worker", 0, "Index of task within the job")
tf.app.flags.DEFINE_boolean("test_mode", False, "Index of task within the job")
tf.app.flags.DEFINE_string("command", 'train', "What the agent should do")
tf.app.flags.DEFINE_integer("start_iteration", -1, "What checkpoint should we use as 'warm start'")

FLAGS = tf.app.flags.FLAGS

sys.path.append(os.path.realpath(".."))
config_name = FLAGS.config

with open('configs/' + config_name, 'r') as fp:
    config = json.load(fp)

algo = hlp.agent_from_config(config)
env = hlp.env_from_config(config)
config['environment'] = env
sess = tf.InteractiveSession()
config['n_features'] = env.get_observation_space()
config['n_actions'] = env.get_action_space()

config['id_worker'] = FLAGS.id_worker
config['test_mode'] = FLAGS.test_mode

agent = algo(sess, config)
if FLAGS.start_iteration >= 0:
    agent.load(config_name[:-5], FLAGS.start_iteration)

method_to_run = getattr(agent, FLAGS.command)
method_to_run()
