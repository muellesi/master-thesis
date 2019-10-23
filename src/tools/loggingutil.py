import logging
import sys


def get_logger(name, level=logging.INFO, do_file_logging=True, file_name=""):
    logger = logging.getLogger(name)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)-15s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(logging.StreamHandler())

    if do_file_logging:
        if file_name == "":
            file_name = "{}.log".format(name)

        logger.addHandler(logging.FileHandler(file_name))

    for handler in logger.handlers:
        handler.level = level
        handler.formatter = formatter

    return logger
