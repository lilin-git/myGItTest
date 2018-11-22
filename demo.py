"""
Demo - how to use PushiPSBot.
"""

import time
from pushi_psbot import PushiPSBot


MODEL_PATH = 'checkpoints/patch_400_sample_10/patch_100_sample_10_pushi_iteration_99000.ckpt'

start_time = time.time()
psbot = PushiPSBot(MODEL_PATH)
print('Initialization took {}'.format(time.time() - start_time))

start_time = time.time()
psbot.enhance('test_images/image1.jpg')
psbot.enhance('test_images/image2.jpg')
psbot.enhance('test_images/image3.jpg')
psbot.enhance('test_images/image4.jpg')
print('Inference took {} for 4 images'.format(time.time() - start_time))




