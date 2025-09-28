#!/bin/zsh

# SAM2
git clone https://github.com/facebookresearch/sam2

cd sam2
cd checkpoints

if [ ! -f sam2_hiera_tiny.pt ]; then
    curl -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
fi

echo "Done!"
echo