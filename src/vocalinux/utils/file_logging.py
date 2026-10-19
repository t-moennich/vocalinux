"""Persistent rotating log file for the vocalinux loggers."""

import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from .paths import logs_dir

LOG_FILE_NAME = "vocalinux.log"

#: Days of daily log files kept besides the current one.
LOG_BACKUP_DAYS = 7

_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"

_file_handler: Optional[logging.Handler] = None


def _install_exception_logging() -> None:
    """Log uncaught exceptions, on any thread, with their traceback."""
    log = logging.getLogger("vocalinux")
    previous_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def log_exception(exc_type, exc_value, exc_traceback) -> None:
        log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        previous_hook(exc_type, exc_value, exc_traceback)

    def log_thread_exception(args) -> None:
        name = args.thread.name if args.thread else "unknown"
        log.critical(
            "Uncaught exception in thread %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        previous_thread_hook(args)

    sys.excepthook = log_exception
    threading.excepthook = log_thread_exception


def setup_file_logging(directory: Optional[str] = None) -> Optional[str]:
    """Log DEBUG and above of the vocalinux loggers to a rotating file.

    The console keeps the level the root logger has now, so the file is more
    detailed than the terminal. Calling it again is a no-op. Returns the log
    path, or None when the file cannot be created.
    """
    global _file_handler
    directory = directory or logs_dir()
    path = os.path.join(directory, LOG_FILE_NAME)
    if _file_handler is not None:
        return path

    try:
        os.makedirs(directory, exist_ok=True)
        handler = TimedRotatingFileHandler(
            path, when="midnight", backupCount=LOG_BACKUP_DAYS, encoding="utf-8"
        )
    except OSError as error:
        logging.getLogger(__name__).warning("Could not open log file %s: %s", path, error)
        return None

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    console_level = root.level
    for existing in root.handlers:
        if existing.level == logging.NOTSET:
            existing.setLevel(console_level)
    root.addHandler(handler)
    # Only the vocalinux loggers go to DEBUG; libraries stay at the root level.
    logging.getLogger("vocalinux").setLevel(logging.DEBUG)
    _file_handler = handler
    _install_exception_logging()
    return path
