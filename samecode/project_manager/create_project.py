import time
import datetime
import os
import argparse
import yaml


def main(args):
    ts = str(datetime.datetime.now())
    path = args.path + '/ID-'+ts.replace(' ', '-').replace(':', '-').replace('-', '-').replace('.', '-')

    os.system('mkdir -p {}/data/'.format(path))
    os.system('mkdir -p {}/code/'.format(path))
    os.system('mkdir -p {}/results/'.format(path))
    os.system('mkdir -p {}/models/'.format(path))
    os.system('mkdir -p {}/environment/'.format(path))


    environment = args.environment
    modeling_framework = args.modeling_framework
    data_modalities = args.data_modalities.split(',')
    cohort = args.cohort.split(',') if args.cohort else []
    feature_representations = args.feature_representations.split(',')
    data_sources = args.data_sources.split(',')

    with open('{}/environment/config.yaml'.format(path), 'w') as file:
        yaml.dump(
            {
                'execution': {
                    'user': os.getlogin(),
                    'environment': environment,
                },
                'model_card': {
                    'version': ts,
                    'modeling_framework': modeling_framework,
                    'data_modalities': data_modalities,
                    'cohort': cohort,
                    'feature_representations': feature_representations,
                    'data_sources': data_sources
                },
            },
            file, default_flow_style=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line for creating the directory structure")

    parser.add_argument(
        '--modeling_framework', 
        type=str, 
        required=True,
        help="Specify the modeling framework (e.g., Clinical Transformer, Bert, GPT, etc.)"
    )

    parser.add_argument(
        '--path', 
        type=str, 
        required=True,
        help="Specify the path were to create the project"
    )

    parser.add_argument(
        '--environment', 
        type=str, 
        required=True,
        help="Specify the environment (container) used for running the modeling"
    )
    
    parser.add_argument(
        '--data_modalities', 
        type=str, 
        required=True, 
        help="Specify the modalities used for the foundation model (e.g., bulk_rna, mutational, copy_number, proteomics). Provide a list separated by spaces"
    )
    
    parser.add_argument(
        '--cohort', 
        type=str, 
        required=False, 
        help="Specify the organ level cohorts used in training (e.g., pan-cancer, cancer-specific, organ-specific cohorts, for multiple cohorts provide a list separated by spaces)."
    )
    
    parser.add_argument(
        '--feature_representations', 
        type=str, 
        required=True,
        help="Specify the number of features used for training the model. For multiple feature representations provie a list separated by spaces"
    )
    
    parser.add_argument(
        '--data_sources', 
        type=str, 
        required=True, 
        help="Specify the number of data sources from which the data was obtained (e.g., TCGA Recount3 GTEx). For multiple data sources representations provie a list separated by spaces"
    )

    args = parser.parse_args()
    main(args)