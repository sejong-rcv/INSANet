#!/usr/bin/env python3
import os
import gdown
import zipfile
import argparse

parser = argparse.ArgumentParser(description='Path to unzip dataset.')

parser.add_argument('--path', type=str,
                    default='./LLVIP')

arg = parser.parse_args()


print('Download LLVIP Dataset (.zip)...')

ggid = "1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"
ggfile = "LLVIP.zip"

gdown.download(f'https://drive.google.com/uc?id={ggid}', output=ggfile)

print('Done. Unzip the dataset.')

with zipfile.ZipFile(ggfile, 'r') as zip_ref:
    zip_ref.extractall(path=arg.path)

os.remove(ggfile)

print(f'Done. Check the path ({arg.path}) to downloaded.')
