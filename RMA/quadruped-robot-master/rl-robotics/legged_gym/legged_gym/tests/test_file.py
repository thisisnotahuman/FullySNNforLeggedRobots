import json
import os

tmp_dict = {'a': 1}
os.mkdir('/home/luo/rl-quadruped/luo')

with open(os.path.join('/home/luo/rl-quadruped/luo', 'env_cfg.json'), 'w') as fp:
    json.dump(tmp_dict, fp)