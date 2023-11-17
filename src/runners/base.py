import logging
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

class BaseRunner:
    def __init__(
            self,
            cfg: DictConfig,
    ):
        self._cfg = cfg
        log.info('run {}'.format(self._cfg['runner']['name']))
        self._output_dir = cfg['output_dir']

    def run(self):
        raise NotImplementedError

