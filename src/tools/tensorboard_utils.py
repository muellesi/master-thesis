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
    __logger.info("Cleaning up tensorboard logdir {}".format(tensorboard_dir))

    run_types = ["train", "validation", "test"]
    for run_type in run_types:
        type_dir = os.path.join(tensorboard_dir, run_type)
        if os.path.exists(type_dir):
            log_files = sorted([os.path.join(type_dir, file) for file in os.listdir(type_dir) if os.path.isfile(os.path.join(type_dir, file))],
                               key=os.path.getmtime)
            if len(log_files) > 0:
                time = datetime.fromtimestamp(os.path.getmtime(log_files[0])).strftime('%Y-%m-%d_%H-%M-%S')
                backupdir = os.path.join(type_dir, "old_runs", time)
                os.makedirs(backupdir)
                for file in log_files:
                    shutil.move(file, backupdir)
                    __logger.info("Moving log file {} to {}".format(file, backupdir))
