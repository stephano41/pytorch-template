import hydra
# import os

@hydra.main(config_path='../conf/', config_name='train')
def main(config):
    print(config)

if __name__ == '__main__':
    main()