import argparse
import sys
import subprocess
import os
from importlib.machinery import SourceFileLoader

torch2trt = SourceFileLoader("torch2trt", "torch2trt/__init__.py").load_module()  # to load relative to root

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--github',
                        type=str,
                        default='https://github.com/NVIDIA-AI-IOT/torch2trt')
    args = parser.parse_args()

    print('| Method | Converter |')
    print('|--------|-----------|')

    for method, entry in torch2trt.CONVERTERS.items():

        if not entry['is_real']:
            continue

        converter = entry['converter']

        # get commit hash
        p = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        commit, err = p.communicate()
        commit = commit.decode('utf-8').strip('\n')

        # get github URL
        url = '{github}/blob/{commit}/{relpath}#L{lineno}'.format(
            github=args.github,
            commit=str(commit),
            relpath=os.path.relpath(converter.__code__.co_filename,
                                    os.path.abspath('.')),
            lineno=converter.__code__.co_firstlineno)

        print('| {method} | [{converter}]({url}) |'.format(
            method=method, converter=converter.__name__, url=url))
