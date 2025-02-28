import time
import datetime
import os
import argparse
import yaml

import logging 

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Command line for creating the directory structure")

    parser.add_argument(
        '--path', 
        type=str, 
        required=True,
        help="Specify a path to create the project"
    )

    args = parser.parse_args()

    ts = str(datetime.datetime.now())
    path = args.path + '/ID-'+ts.replace(' ', '-').replace(':', '-').replace('-', '-').replace('.', '-')

    logger.info(f'project directory: {path}')
    logger.info('''
    under ./environment/ make sure to update the yaml file with the necessary information
    to reproduce the analysis.
    ''')
    os.system('mkdir -p {}/data/'.format(path))
    os.system('mkdir -p {}/code/'.format(path))
    os.system('mkdir -p {}/results/'.format(path))
    os.system('mkdir -p {}/models/'.format(path))
    os.system('mkdir -p {}/environment/'.format(path))

    with open('{}/environment/config.yaml'.format(path), 'w') as file:
        yaml.dump(
            {
                'execution': {
                    'environment': None,
                },
                'model_card': {
                    'version': ts,
                    'framework': None,
                    'data': None,
                    'data_sources': None
                },
            },
            file, default_flow_style=False
        )

if __name__ == "__main__":
    main()