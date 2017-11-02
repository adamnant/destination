"""
configuration loader
"""

# library imports
import os
import funconf

local_conf = os.path.join(os.path.expanduser('~'), '.destinations.conf')
default_conf = os.path.join(os.path.dirname(__file__), 'default.conf')

# config priority is user's home directory > default confs
paths = [default_conf, local_conf]

# make the config object
config_obj = funconf.Config(paths)

# if the local_conf doesn't already exist, make it
if not os.path.exists(local_conf):
    with open(local_conf, 'wb') as f:
        f.write(str(config_obj))
