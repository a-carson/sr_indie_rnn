import logging
import os
from typing import Optional
import yaml
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback, Namespace, Trainer, LightningModule, get_filesystem
from pytorch_lightning.loggers import WandbLogger


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

LOG_PATH = os.environ.get('WANDB_LOG_DIR', default='wandb_logs/')
CONFIG_DIR = 'configs/'
CLI_CONFIG_NAME = 'cli_config.yaml'
CONFIG_SAVE_NAME = 'pl_config.yaml'

class CustomSaveConfigCallback(SaveConfigCallback):

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = CONFIG_SAVE_NAME,
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.already_saved = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if len(trainer.loggers) > 1:
            log_dir = trainer.loggers[1].experiment.dir
        else:
            log_dir = trainer.log_dir  # this broadcasts the directory

        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)


class CustomLightningCLI(LightningCLI):
    def __init__(self, cli_config_path: Optional[str] = None, *args, **kwargs) -> None:
        if cli_config_path is None:
            cli_config_path = os.path.join(CONFIG_DIR, CLI_CONFIG_NAME)
        assert os.path.isfile(cli_config_path)
        with open(cli_config_path, "r") as in_f:
            self.cli_config = yaml.safe_load(in_f)
        super().__init__(*args, **kwargs, save_config_callback=CustomSaveConfigCallback)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for add_arg in self.cli_config["additional_arguments"]:
            name = add_arg["name"]
            default = add_arg.get("default")
            if default is not None:
                parser.add_argument(name, default=default)
            else:
                parser.add_argument(name)

        for link_args in self.cli_config["link_arguments"]:
            parser.link_arguments(link_args["src"], link_args["dest"])


    def before_fit(self) -> None:

        if self.config.fit.custom.use_wandb:
            path = os.path.join(LOG_PATH, self.config.fit.custom.project_name)
            if not os.path.exists(path):
                os.mkdir(path)

            name = self.config.fit.custom.name
            if name == "":
                name = None

            wandb_logger = WandbLogger(project=self.config.fit.custom.project_name,
                                       log_model=True,
                                       save_dir=path,
                                       group=self.config.fit.custom.experiment_name,
                                       name=name)
            self.trainer.loggers.append(wandb_logger)

            # modify checkpoint path
            ckpt_path = os.path.join(wandb_logger.experiment.dir, 'checkpoints')
            self.trainer.checkpoint_callback.dirpath = ckpt_path

            log.info("wand enabled")

        else:
            log.info("wandb is disabled")


    def after_fit(self) -> None:
        self.trainer.test(ckpt_path='last', dataloaders=self.datamodule.test_dataloader())