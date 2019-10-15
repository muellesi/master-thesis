# -*- coding: utf-8 -*-

import logging
import urllib
import os
import shutil

dataUrls = [

        ]

dataDirName = "data"
dataDirDownloadCache = "cache"

def get_logger():
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s', level=logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s: %(message)s')
    for handler in logger.handlers:
        handler.formatter = formatter
    return logger

def get_data():
    global dataDirName
    dataDirName = os.path.abspath(dataDirName)
    logger = get_logger();

    for url in dataUrls:
        logger.info("Downloading {}".format(url))    
        
        urlFileName = os.path.basename(url)
        
        cachePath = os.path.join(dataDirName, dataDirDownloadCache)
        if not os.path.exists(cachePath):
            os.makedirs(cachePath)
            
        downloadedFile = os.path.join(cachePath, urlFileName)
        
        if not os.path.exists(downloadedFile):
            urllib.request.urlretrieve(url, downloadedFile)
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


if __name__ == "__main__":
    get_data()