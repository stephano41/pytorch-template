import logging
import os

import hydra
from hydra.utils import instantiate

logger = logging.getLogger(__name__)

# when on windows, scipy bug causes ray tune to not save trials properly, see https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
if os.name == 'nt':
    import _thread
    import win32api


    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        if dwCtrlType == 0:  # CTRL_C_EVENT
            hook_sigint()
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler


    win32api.SetConsoleCtrlHandler(handler, 1)


@hydra.main(config_path='conf', config_name='tune', version_base='1.3')
def main(config):
    instantiate(config.pipeline, config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
