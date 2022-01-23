#!/usr/bin/env bash

# in ascending order of effort required to obtain them
declare -a newsoutlets
newsoutlets=("vox.com" "formula1.com" "wired.com" "bostonglobe.com" "theweek.com" "lawandcrime.com" "vanityfair.com" "seattletimes.com" "politico.com" "technologyreview.com" "newrepublic.com" "theverge.com" "avclub.com" "rollingstone.com" "espn.com" "thehill.com" "independent.co.uk" "deadspin.com" "usatoday.com" "wsj.com" "newsweek.com" "kotaku.com" "metro.us" "denverpost.com" "theepochtimes.com" "abcnews.go.com" "gizmodo.com" "cnn.com" "nbc.com" "apnews.com" "washingtonpost.com" "bbc.com" "theguardian.com" "mirror.co.uk" "cnbc.com" "huffpost.com" "businessinsider.com" "foxnews.com" "bleacherreport.com" "marketwatch.com")

for outlet in "${newsoutlets[@]}"; do
  python3 run_pipeline.py run \
    resources/yaml/device_docker.yaml \
    resources/yaml/article_extraction/base.yaml \
    resources/yaml/article_extraction/english/${outlet}.yaml
done