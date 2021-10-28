import logging
from rich.logging import RichHandler

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
    ],
)
WE_LOGGER = logging.getLogger(__name__)
