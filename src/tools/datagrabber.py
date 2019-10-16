# -*- coding: utf-8 -*-
import urllib
import os
import shutil
from progressbar import ProgressBar
from loggingutil import get_logger

dataDirName = r"C:\data"
dataDirDownloadCache = "cache"

pbar = None
downloaded = 0

def show_download_progress(count, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = ProgressBar(maxval=total_size)
    global downloaded
    downloaded += block_size
    if downloaded > total_size:
        downloaded = total_size
    pbar.update(downloaded)
    if downloaded == total_size:
        pbar.finish()
        pbar = None
        downloaded = 0

def get_data(dataUrls):
    global dataDirName
    dataDirName = os.path.abspath(dataDirName)
    logger = get_logger(__file__);

    for url in dataUrls:
        logger.info("Downloading {}".format(url))    
        
        urlFileName = os.path.basename(url)
        
        cachePath = os.path.join(dataDirName, dataDirDownloadCache)
        if not os.path.exists(cachePath):
            os.makedirs(cachePath)
            
        downloadedFile = os.path.join(cachePath, urlFileName)
        
        if not os.path.exists(downloadedFile):
            urllib.request.urlretrieve(url, downloadedFile, show_download_progress)
        else:
            logger.warning("Target path {} already exists! Using cached version.".format(downloadedFile))
        
        fileNameWithoutExtension = os.path.splitext(urlFileName)[0]
        
        archiveExts = ['.tar', '.gz', '.bz2', '.zip']

        isArchive = False
        for ext in archiveExts:
            if downloadedFile.endswith(ext):
                isArchive = True
                break
            
        if isArchive:
            logger.info("Extracting {} to {}".format(downloadedFile, os.path.join(dataDirName, fileNameWithoutExtension)))
            shutil.unpack_archive(downloadedFile, os.path.join(dataDirName, fileNameWithoutExtension))


def read_data_urls():
    dataUrls = []
    with open("../datasets.txt", 'r') as file:
        dataUrls = file.readlines()
    # allow to comment out data urls
    dataUrls = [url for url in dataUrls if not (url.startswith("#") or url.startswith("//"))]
    return dataUrls    
        

if __name__ == "__main__":
    dataUrls = read_data_urls()
    get_data(dataUrls)