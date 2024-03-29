#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

dataset_root=${1}

if [ "$#" -ne 1 ]; then
    echo "Usage: ./download_dataset.sh DATASET_DOWNLOAD_PATH"
    exit 2
fi

# Check if directory exists
if [[ ! -d "${dataset_root}" ]]
then	
	echo "Invalid destination path: ${dataset_root}."	
	exit 2
fi

echo ""
echo "Downloading dataset to ${dataset_root}"
echo ""

url="https://onedrive.live.com/download?cid=1570430EADF56512&resid=1570430EADF56512%21109419&authkey=AJcMP-7Yp86PWoE"
wget --no-check-certificate ${url} -O ${dataset_root}/kaist-cvpr15.tar.gz

echo ""
echo "Extract dataset (takes > 10 mins)"
echo ""
tar zxvf ${dataset_root}/kaist-cvpr15.tar.gz >/dev/null 2>&1 && rm ${dataset_root}/kaist-cvpr15.tar.gz

echo ""
echo "Add a symbolic link"
echo ""
ln -s ${dataset_root} kaist-rgbt

echo ""
echo "Done."
echo ""
