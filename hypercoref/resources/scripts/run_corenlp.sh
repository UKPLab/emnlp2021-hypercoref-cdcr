#!/usr/bin/env bash
java -Xmx16G -cp "/.../stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port $1 -timeout 600000 -threads $2 -maxCharLength 500000 -quiet True -preload ssplit,tokenize,pos,lemma,ner,depparse