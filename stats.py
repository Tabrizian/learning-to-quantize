import os
import re
import argparse
import numpy as np


def args():
    parser = argparse.ArgumentParser(description='Check runs status')
    parser.add_argument('--pattern', type=str)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--runs', action='store_true')
    return parser.parse_args() 

def print_logs(outputs, opt):
    sorted_index = np.argsort(outputs['number'])
    if opt.show:
        for index in sorted_index:
            if opt.runs:
                print('===========', outputs['number'][index],
                        outputs['date'][index], outputs['run_names'][index])
                continue
            else:
                print('===========', outputs['number'][index],
                        outputs['date'][index])
            if len(outputs['logs'][index]) > 0:
                print(outputs['logs'][index])
            else:
                print('No log exist')
    else:
        for index in sorted_index:
            print(outputs['number'][index], outputs['date'][index])


if __name__ == '__main__':
    opt = args()
    logdir = 'runs'
    pattern = opt.pattern
    num = opt.num

    run_names = []
    for root, subdirs, files in os.walk(logdir, followlinks=True):
        if re.match(pattern, root):
            run_names += [root]
    outputs = {
            'number': [],
            'date': [],
            'logs': [],
            'run_names': []
            }
    for run_name in run_names:
        number = run_name.split('/')[-1].split('_')[1]
        outputs['number'].append(number)
        f = open(run_name + '/log.txt', 'r')
        lines = f.read().splitlines()
        lines.reverse()
        f.close()

        for last_line in lines:
            if len(last_line.split(' ')) > 1 and last_line.split(' ')[0].startswith('2020'):
                last_line = last_line.split(' ')
                break

        if len(last_line) > 1:
            outputs['date'].append(' '.join([last_line[0], last_line[1]]))
        else:
            outputs['date'].append('No Date')

        f = open(run_name + '/log', 'r')
        lines = f.read().splitlines()
        outputs['logs'].append('\n'.join(lines[-num:]))
        outputs['run_names'].append(run_name)
        f.close()

    print_logs(outputs, opt)
