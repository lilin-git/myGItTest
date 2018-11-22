"""
Demo - how to use PushiPSBot.
"""

import time
from pushi_psbot import PushiPSBot


MODEL_PATH = 'checkpoints/patch_400_sample_10/patch_100_sample_10_pushi_iteration_99000.ckpt'

start_time = time.time()
psbot = PushiPSBot(MODEL_PATH)
print('Initialization took {}'.format(time.time() - start_time))





