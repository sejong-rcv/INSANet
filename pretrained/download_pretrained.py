#!/usr/bin/env python3

import gdown

print('Download checkpoint... ')

ggid = "1iP1XvRu0mvcGLSidY9ML0tVRUJrdPcRm"
ggfile = "pretrained.pth.tar"

gdown.download(f'https://drive.google.com/uc?id={ggid}', output=ggfile)

print('Done. Please try "python src/eval.py --model pretrained/pretrained.pth.tar"')