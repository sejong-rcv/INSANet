#!/usr/bin/env python3
import gdown
import shutil

print('Download checkpoint... ')

ggid = "1C56Jq1K2TuXFAp9f5UDkSF7Y-FucAG0L"
ggfile = "pretrained.pth.tar"

gdown.download(f'https://drive.google.com/uc?id={ggid}', output=ggfile)

try:
    shutil.move(ggfile, './pretrained/pretrained.pth.tar')
except FileNotFoundError:
    pass

print('Done. Please try "python src/script/inference.sh"')
