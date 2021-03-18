"""
This files purpose is to be used as a script to create variants of yaml config.

It takes the initial config file as input and the number of reruns to be done per experiment.

The syntax introduced by which new copies are made are fields with a suffix '_set',
which is always followed by a list.

So usually we might have a parameter in our config:

lr: .1

Now we can have:

lr_set:
    - .1
    - .2
    - .4

This will now yield the file being copied 3 times with lr_set replaced by the respective lr settings, like:
lr: .4

So every field with a '_set' has to have a list as value and for each entry in each of these lists a copy will be made,
s.t. if we had two '_set' fields with 2 and 5 entries respectively we would get 2*5=10 files.

These files will be saved in the same directory as the input config with name changes like _opt.lr=.2_1try.
"""

import argparse
import yaml
from copy import deepcopy


def access_with_path(dict_or_list, list_of_indices):
    current_dict_or_list = dict_or_list
    for idx in list_of_indices:
        current_dict_or_list = current_dict_or_list[idx]
    return current_dict_or_list

def find_all_fields_with_suffix(dict_or_list,suffix):
    result_paths = []
    if isinstance(dict_or_list,dict):
        keys = dict_or_list.keys()
        for key in keys:
            if key.endswith(suffix):
                result_paths.append([key])

    key_iterator = range(len(dict_or_list)) if isinstance(dict_or_list, list) else dict_or_list.keys()

    for key in key_iterator:
        if isinstance(dict_or_list[key], list) or isinstance(dict_or_list[key], dict):
            sub_result_sub_paths = find_all_fields_with_suffix(dict_or_list[key],suffix)
            result_paths += [[key] + p for p in sub_result_sub_paths]

    return result_paths

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('number_of_reruns', type=int)
    args = ap.parse_args()

    with open(args.config,'r') as config_file:
        initial_config = yaml.load(config_file,yaml.SafeLoader)

    set_paths = find_all_fields_with_suffix(initial_config, '_set')

    name_prefix = args.config.replace('.yaml','')

    config_set = [(initial_config,name_prefix)]

    for set_path in set_paths:
        options = access_with_path(initial_config, set_path)
        new_config_set = []
        for config, name_prefix in config_set:
            for option in options:
                config_copy = deepcopy(config)
                mother_dict = access_with_path(config_copy, set_path[:-1])
                mother_dict.pop(set_path[-1])
                mother_dict[set_path[-1][:-len('_set')]] = option
                new_config_set.append((config_copy,name_prefix+'__'+'.'.join(set_path)[:-len('_set')]+'='+str(option)))
        config_set = new_config_set


    for config, path_prefix in config_set:
        for rerun_idx in range(1, args.number_of_reruns + 1):
            filename = path_prefix+f'__{rerun_idx}try.yaml'
            print(filename)
            with open(filename, 'w') as config_file:
                yaml.dump(config,config_file)




