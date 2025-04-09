import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from pixelcache.tools.logging import DEFAULT_VERBOSITY, LoggingRich
from pixelcache.tools.utils import seed_everything

load_dotenv()

SEED_EVERYTHING = os.getenv("SEED_EVERYTHING", "-1")
CUDA_DETERMINISTIC = os.getenv("CUDA_DETERMINISTIC", "False")
if SEED_EVERYTHING != "-1":
    seed_everything(
        int(SEED_EVERYTHING),
        verbose=False,
        cuda_deterministic=CUDA_DETERMINISTIC == "True",
    )


@lru_cache(maxsize=128)
def get_logger(verbosity: dict[str, bool] = DEFAULT_VERBOSITY) -> LoggingRich:
    """Create a LoggingRich object with a specified verbosity level."""
    return LoggingRich(verbosity=verbosity)


def wait_seconds_bar(total_seconds: int, /, description: str) -> None:
    """Display a progress bar for a specified number of seconds."""
    # rich progress bar with time
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=Console(),
    ) as progress:
        task = progress.add_task(description, total=total_seconds)
        for _ in range(total_seconds):
            time.sleep(1)
            progress.update(task, advance=1)
