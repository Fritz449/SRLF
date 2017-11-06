import tensorflow as tf
import numpy as np
import os
import sys
import random
import subprocess
from redis import Redis

sys.path.append(os.path.realpath(".."))

import helpers.utils as hlp