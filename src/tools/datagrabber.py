# -*- coding: utf-8 -*-
import urllib
import os
import shutil
from progressbar import ProgressBar
from loggingutil import get_logger

dataDirName = r"C:\data"
dataDirDownloadCache = "cache"

progress_bar = None
downloaded = 0


def show_download_progress(count, block_size, total_size):
    global progress_bar
    if progress_bar is None:
        progress_bar = ProgressBar(maxval=total_size)
    global downloaded
    downloaded += block_size
    if downloaded > total_size:
        downloaded = total_size
    progress_bar.update(downloaded)
    if downloaded == total_size:
        progress_bar.finish()
        progress_bar = None
        downloaded = 0


def get_data(data_urls):
    global dataDirName
    dataDirName = os.path.abspath(dataDirName)
    logger = get_logger(__file__)

    for url in data_urls:
        logger.info("Downloading {}".format(url))

        url_file_name = os.path.basename(url)

        cache_path = os.path.join(dataDirName, dataDirDownloadCache)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        downloaded_file = os.path.join(cache_path, url_file_name)

        if not os.path.exists(downloaded_file):
            urllib.request.urlretrieve(url, downloaded_file, show_download_progress)
        else:
            logger.warning("Target path {} already exists! Using cached version.".format(downloaded_file))

        file_name_without_extension = os.path.splitext(url_file_name)[0]

        archive_exts = ['.tar', '.gz', '.bz2', '.zip']

        is_archive = False
        for ext in archive_exts:
            if downloaded_file.endswith(ext):
                is_archive = True
                break

        if is_archive:
            logger.info(
                "Extracting {} to {}".format(downloaded_file, os.path.join(dataDirName, file_name_without_extension)))
            shutil.unpack_archive(downloaded_file, os.path.join(dataDirName, file_name_without_extension))


def read_data_urls():
    data_urls = []
    with open("../datasets.txt", 'r') as file:
        data_urls = file.readlines()
    # allow to comment out data urls
    data_urls = [url for url in data_urls if not (url.startswith("#") or url.startswith("//"))]
    return data_urls


if __name__ == "__main__":
    Urls = read_data_urls()
    get_data(Urls)
