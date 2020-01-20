# -*- coding: utf-8 -*-
import os
import shutil
import urllib

from progressbar import ProgressBar

from tools import get_logger



downloaded = 0
cache_path = "C:\\temp\\"


def show_download_progress(count, block_size, total_size, progress_bar):
    if progress_bar is None:
        progress_bar = ProgressBar(maxval = total_size)
    global downloaded
    downloaded += block_size
    if downloaded > total_size:
        downloaded = total_size
    progress_bar.update(downloaded)
    if downloaded == total_size:
        progress_bar.finish()
        downloaded = 0


def download_data(url, target_folder = cache_path):
    logger = get_logger(__file__)

    logger.info("Downloading {}".format(url))

    url_file_name = os.path.basename(url)
    file_name_without_extension = os.path.splitext(url_file_name)[0]

    target_folder = os.path.join(target_folder, file_name_without_extension)

    os.makedirs(target_folder, exist_ok = True)

    downloaded_file = os.path.join(target_folder, url_file_name)

    if not os.path.exists(downloaded_file):
        pbar = None
        urllib.request.urlretrieve(url, downloaded_file,
                                   lambda count, block_size, total_size: show_download_progress(count, block_size,
                                                                                                total_size, pbar))
        del pbar
    else:
        logger.warning("Target path {} already exists! Using cached version.".format(downloaded_file))

    archive_exts = ['.tar', '.gz', '.bz2', '.zip']

    is_archive = False
    for ext in archive_exts:
        if downloaded_file.endswith(ext):
            is_archive = True
            break

    if is_archive:
        logger.info(
                "Extracting {} to {}".format(downloaded_file,
                                             os.path.join(target_folder, file_name_without_extension)))
        shutil.unpack_archive(downloaded_file, os.path.join(target_folder, file_name_without_extension))
