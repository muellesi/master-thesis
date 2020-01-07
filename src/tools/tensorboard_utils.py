import os
from datetime import datetime
import shutil
import tools



__logger = tools.get_logger(__name__, do_file_logging=False)


def clean_tensorboard_logs(tensorboard_dir):
    """
    Moves files in subdirs "train", "validation" and "test" of tensorboard_dir to a new subdir "old_runs". Creates timestamped subdirs.
    :param tensorboard_dir: Root tensorboard log directory.
    """
    if os.path.exists(tensorboard_dir):
        __logger.info("Cleaning up tensorboard logdir {}".format(tensorboard_dir))
        log_files = sorted([os.path.join(tensorboard_dir, file) for file in os.listdir(tensorboard_dir)],
                                   key=os.path.getmtime)
        if len(log_files) > 0:
            backupdir = os.path.abspath(os.path.join(tensorboard_dir, os.pardir))
            backupdir = os.path.join(backupdir, "tensorboard_old_runs")
            if not os.path.exists(backupdir):
                os.makedirs(backupdir)

            time = datetime.fromtimestamp(os.path.getmtime(log_files[0])).strftime('%Y-%m-%d_%H-%M-%S')
            backupdir = os.path.join(backupdir, time)

            __logger.info("Moving old log files from {} to {}".format(tensorboard_dir, backupdir))
            os.rename(tensorboard_dir, backupdir)

    os.makedirs(tensorboard_dir)
