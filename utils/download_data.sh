# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Download everything
wget --show-progress -O data/attr-ops-data.tar.gz https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops-data.tar.gz
wget --show-progress -O data/mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
wget --show-progress -O data/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
wget --show-progress -O data/splits.tar.gz http://www.cs.cmu.edu/~spurushw/publication/compositional/compositional_split_natural.tar.gz

echo "Data downloaded. Extracting files..."

# Dataset metadata, pretrained SVMs and features, tensor completion data
tar -zxvf data/attr-ops-data.tar.gz --strip 1

# dataset images
cd data/

# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
rename "s/ /_/g" mit-states/images/*

# UT-Zappos50k
unzip utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/_images/

# Download new splits for Purushwalkam et. al
tar -zxvf splits.tar.gz

cd ..
python utils/reorganize_utzap.py

# remove all zip files and temporary files
rm -r data/attr-ops-data.tar.gz data/mitstates.zip data/utzap.zip data/ut-zap50k/_images data/splits.tar.gz

