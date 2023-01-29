import tdg.meta
from tdg.license import license
import argparse

def print_and_exit(description, data):

    class anonymous(argparse.Action):

        def __call__(self, parser, namespace, values, option_string):
            print(data)
            exit()

    return {
            'help':   f'Exit after printing {description}.',
            'action': anonymous,
            'nargs':  0,
            }

def defaults():
    meta_arguments = argparse.ArgumentParser(add_help=False)
    meta_arguments.add_argument('--version', **print_and_exit("the version", tdg.meta.version))
    meta_arguments.add_argument('--license', **print_and_exit("the license", license))

    return meta_arguments

if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[defaults(), ]) 
    args = parser.parse_args()

