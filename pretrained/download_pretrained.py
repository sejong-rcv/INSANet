#!/usr/bin/env python3

import gdown

print('Download checkpoint... ')

ggid = "1yL6H0x8pTaOqmZ6U05kFiTHfx_MO6YdW"
ggfile = "pretrained.pth.tar"

gdown.download(f'https://drive.google.com/uc?id={ggid}', output=ggfile)

print('Done. Please try "python src/script/inference.sh"')
