from tensorboard.backend.event_processing import event_accumulator
from os import listdir
from os.path import isfile, join
import re
import argparse
import numpy as np, scipy.stats as st
def get_last_metric(path, metric, get_step=None):
    onlyfiles = sorted([join(path,f) for f in listdir(path) if isfile(join(path, f))])
    last_point = 0
    v = None
    for f in onlyfiles:
        ea = event_accumulator.EventAccumulator(f)
        # top1 not found
        ea.Reload()
        try:
            if get_step is not None:
                for e in ea.Scalars(metric):
                    last_point = e.step
                    v = e.value
                    if get_step is not None and last_point == get_step:
                        return v, last_point
            e = ea.Scalars(metric)[-1]
            if e.step >= last_point:
                if last_point > 0:
                    print("Warning: Multiple runs with one name:", f, "other result:", v, 'at', last_point)
                last_point = e.step
                v = e.value
        except Exception as e:
            print(e)
    return v, last_point

def get_results(logdir, mypath, split='test', metric='top1', assert_step=None):
    mypath = mypath.split('/')[-1]
    suffix = '.yaml'
    if mypath.endswith(suffix):
        mypath = mypath[:-len(suffix)]
    paths = [path for path in listdir(logdir) if re.search(f'{mypath}(_[0-9]+try|).yaml', path)]
    print([path[len(mypath):] for path in paths])

    paths = [join(join(logdir, path), split) for path in paths]
    results = [get_last_metric(path, metric, get_step=assert_step) for path in paths]
    assert all([r[1] == results[0][1] for r in results]), results
    step = results[0][1]
    results = [r[0] for r in results]
    assert all(r is not None for r in results), results
    if assert_step is not None:
        assert assert_step == step, f'{assert_step} vs {step}'
    print(f"The results are the following {len(results)} at step {step}: {results}")

    results = np.array(results)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Top1 Results.')
    parser.add_argument('path',
                   help='Prefix of the paths/conf files for the runs to evaluate.')
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--split', default='test')
    parser.add_argument('--metric', default='top1', help='Can be e.g. top1, top5, loss, eval_top1')
    parser.add_argument('--step', default=None, type=int)
    args = parser.parse_args()
    mypath = args.path.split('/')[-1]
    results = get_results(args.logdir,args.path,args.split,args.metric, assert_step=args.step)
    n = len(results)
    m, se = np.mean(results), st.sem(results)
    confidence = .95
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    print(f"Mean: {round(np.mean(results),4)}, Std: {round(np.std(results),4)}, +/-: {round(h,4)}")
    print(f"{round(np.mean(results)*100,2)} $\pm$ {str(round(h,4)*100)[1:]}")
