import logging
import sys
from vehicles_model.config import config

VERSION_PATH = config.PACKAGE_ROOT / 'VERSION'

with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()


FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


# Configure logger for use in package
logger = logging.getLogger('vehicles_model')
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())
logger.propagate = False
