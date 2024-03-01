# Get the dataset ready
We thank Hwang and Jia, who are the authors for providing valuable datasets.([KAIST](https://soonminhwang.github.io/rgbt-ped-detection/), [LLVIP](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)).

## KAIST Multispectral Pedestrian Detection Dataset
Please run the script to download KAIST dataset from one drive (36GB)
```
$ sh download_kaist.sh $PATH_TO_DOWNLOAD
(e.g) sh download_kaist.sh ./
```

---

## LLVIP: A Visible-infrared Paired Dataset for Low-light Vision
Please run the python file (.py) to download LLVIP dataset from google drive (4GB)
```
python download_llvip.py --path=PATH_TO_DOWNLOAD
(e.g.) python download_llvip.py --path='./'
```
