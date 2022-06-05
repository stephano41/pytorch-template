import hydra
# import os
from utils import instantiate


@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    metric = [instantiate(met, is_func=True) for met in config['metrics']]
    print(metric)

if __name__ == '__main__':
    main()