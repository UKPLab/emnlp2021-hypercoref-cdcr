#!/usr/bin/env bash

URL="https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3390/hypercoref_page_index_filtered.7z"
DESTINATION="/hypercoref/working_dir/page_seeding/"

mkdir -p $DESTINATION
cd $DESTINATION
wget -c $URL
7z x hypercoref_page_index_filtered.7z
if [ $? -eq 0 ]; then
  rm hypercoref_page_index_filtered.7z
  echo "All files extracted."
fi