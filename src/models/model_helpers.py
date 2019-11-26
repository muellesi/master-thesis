import os
import tools



__logger = tools.get_logger(__name__, do_file_logging=False)


def try_load_checkpoint(model, checkpoint_dir: str, checkpoint_file_prefix: str = ""):
    """
    Tries to load a checkpoint file for the specified model
    :param checkpoint_dir: directory where the checkpoints are saved
    :return: Model with loaded checkpoint if checkpoint existed, else unmodified model
    """
    if os.path.exists(checkpoint_dir):
        cp_files = [os.path.abspath(os.path.join(checkpoint_dir, filename)) for filename in os.listdir(checkpoint_dir)]
        cp_files = [path for path in cp_files if os.path.isfile(path) and checkpoint_file_prefix in os.path.basename(path)]

        if len(cp_files) > 0:
            files_sorted = sorted(cp_files, key=os.path.getctime, reverse=True)
            for latest_file in files_sorted:
                try:
                    __logger.info("Trying to load weights from {}".format(latest_file))
                    model.load_weights(latest_file)
                    __logger.info("Loading successful!")
                    return model
                except Exception as e:
                    __logger.exception(e)
                    __logger.error("Loading of {} failed!".format(latest_file))

    return model
