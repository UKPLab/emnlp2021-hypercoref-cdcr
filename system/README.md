# ariecattan/coref refreshed
Refreshed version of [ariecattan/coref](https://github.com/ariecattan/coref) used for experiments in our EMNLP 2021 paper.

- [Notable changes](#notable-changes)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Usage (Coreference Resolution)](#usage-coreference-resolution)
  - [Training](#training)
  - [Hyperparameter optimization](#hyperparameter-optimization)
  - [Predicting on Test Splits](#predicting-on-test-splits)
  - [Scoring](#scoring)
- [Usage (Event Extraction)](#usage-event-extraction)
  - [Training](#training-1)
  - [Predicting](#predicting)
  - [Scoring](#scoring-1)
- [Original Project Readme](#cross-document-coreference-resolution)

## Notable changes
* More efficient parallelized pytorch dataloader:
  * With the original implementation, GPUs would run at ~20% load. Support for the e2e-coref style mention generation was not needed in our experiments and was dropped. A stub method `_generate_and_prune_mentions` exists in [`train_pairwise_scorer_mb.py`](train_pairwise_scorer_mb.py) should anyone need to reimplement this feature in the future.
* Documents are transformer-embedded just in time before the optimizer step:
  * Previously, training the model on one topic would transformer-embed all documents in that topic and store their embeddings on the GPU, which does not scale for topics beyond ECB+ size. The implementation could not afford fine-tuning of transformer weights for this reason. (Trainable transformer weights would be possible now, though we decided against this to keep results comparable to the original implementation.)
* Added early stopping and warm-starting from previous checkpoints.
* Simplifications and more code comments throughout.

## Requirements
- python3 >= 3.7.12
- `pip install -r requirements.txt`, this installs torch 1.7.1 with CUDA 10.2
  - for CUDA 11, run `pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html`

## Data Preparation
Follow [this project](https://github.com/UKPLab/emnlp2021-hypercoref-cdcr/tree/master/hypercoref) to obtain the ECB+, FCC-T, GVC and (parts of the) HyperCoref corpora in the right format.

Then, sort the files into the pre-existing directory structure at `data/`. The JSON config files under [`configs/`](configs/) expect the following directory structure for each corpus (here, `S_gg` ECB+):
```
data/gg/ecbp
├── ...
├── gold
│   ├── dev_events_corpus_level.conll
│   └── test_events_corpus_level.conll
└── mentions
    ├── dev_events.json
    ├── dev.json
    ├── test_events.json
    ├── test.json
    ├── train_events.json
    └── train.json
```

## Usage (Coreference Resolution)
Running the coreference resolution system entails four main steps:
1. Training
2. Optimizing the agglomerative clustering threshold on the development split:
   1. Predicting on the development split, which produces CoNLL files
   2. Scoring models with coreference resolution metrics
3. Predicting on the test split to produce CoNLL files
4. Scoring models with coreference resolution metrics

We outline the steps for `S_gg` ECB+. To run experiments for other experimental scenarios or corpora, use the other JSON configuration files inside `configs/`.

For the paper experiments, we ran the above steps three separate times with different random seeds.
We use random seed `0` for our running example. The random seed can be specified by changing the `"random_seed"` setting inside each of the JSON config files.

### Training
Run:
```bash
python3 train_pairwise_scorer_mb.py train configs/coreference/gg/ecbp/config_gg_ecbp_train.json
```
This will produce several model checkpoint files inside `models/gg/ecbp/events/0/`.

### Hyperparameter optimization
Take note of the epoch with the lowest dev-loss. Copy its model checkpoints to `models/gg/ecbp/events/0/best` (`0` being the random seed in the path), like so:
```
models/gg/ecbp/events/0/best
├── pairwise_scorer_0
├── span_repr_0
└── span_scorer_0
```

Then, predict on the dev split using several different agglomerative clustering thresholds:
```bash
python3 predict_tune_mb.py tune configs/coreference/gg/ecbp/tune/config_gg_ecbp_tune_seed0.json
```
This produces CoNLL files under `models/gg/ecbp/events/0/tune_on_best/best`.

To determine the clustering threshold that produces the best LEA F1 scores, run:
```bash
python3 run_scorer_mb.py \
        data/gg/ecbp/gold/dev_events_corpus_level.conll \
        models/gg/ecbp/events/0/tune_on_best/best
```

### Predicting on Test Splits
Consider setting the best-performing epoch and clustering threshold per seed in the JSON config:
  * For ECB+ `S_gg`, these are `configs/coreference/gg/ecbp/config_gg_ecbp_test_ecbp_closure.json` and `configs/coreference/gg/ecbp/config_gg_ecbp_test_ecbp_predicted.json`. (The two files differ in the type of topic pre-clustering used.)
  * Towards the bottom, there is a setting `best_hyperparams` which expects the following schema:
    ```json
    {
      "SEED": [BEST_EPOCH, CLUSTERING_THRESHOLD],
      ...
    }
    ```
    Note: The prediction script can predict for multiple random seeds in a single execution.

To create test split predictions (in this example, using a config for ECB+ with the gold-standard topic preclustering), run:
```bash
python3 predict_tune_mb.py predict configs/coreference/gg/ecbp/config_gg_ecbp_test_ecbp_closure.json
```

The resulting CoNLL file is located at `models/gg/ecbp/events/closure/0/model_0_test_events_average_THRESHOLD_corpus_level.conll`

### Scoring
Due to a bug (that I was too lazy to fix), all CoNLL prediction files start with `#begin document dev_events` even when they contain test split data. This is easily fixed with `sed`:
```bash
find models/ -name "*test_events*.conll" -exec sed -i 's/document dev_events/document test_events/g' {} +
```

To compute and aggregate the final coreference metrics (for the running example of `S_gg` ECB+), run:
```bash
python3 aggregate_scores_mb.py \
        data/gg/ecbp/gold/test_events_corpus_level.conll \
        models/gg/ecbp/events/closure
```

Afterwards, `models/gg/ecbp/events/closure/scores.txt` will contain the metrics in CSV format and pretty-printed. `scores.pkl` in the same directory contains the pickled pandas dataframe for later analyses.

## Usage (Event Extraction)
We ran event extraction against the ECB+ test split, re-using the hyperparameters and the original training loop from Cattan et al.
We outline the steps for `S_gg` ECB+ using random seed `0`.

### Training
Run:
```bash
python3 train_pairwise_scorer.py train configs/event_extraction/gg/config_gg_train.json
```
This will produce several model checkpoint files inside `models/event_extraction/gg/ecbp/events_10_0.25/`.

### Predicting
Copy the model checkpoints of the epoch with lowest dev loss to `models/event_extraction/gg/ecbp/events_10_0.25/0/best` (`0` being the random seed in the path), like so:
```
models/event_extraction/gg/ecbp/events_10_0.25/0/best
├── pairwise_scorer_0
├── span_repr_0
└── span_scorer_0
```

Run:
```bash
python3 predict.py predict configs/event_extraction/gg/config_gg_test_seed0.json
```

This produces a CoNLL file at `models/event_extraction/gg/ecbp/events_10_0.25/ecbp/closure/0/test_events_average_1e-05_model_0_corpus_level.conll`

### Scoring
Firstly, make sure to fix the CoNLL file header as in the coreference resolution example:
```bash
find models/event_extraction -name "*.conll" -exec sed -i 's/test_events_average_1e-05_model_0/test_events/g' {} +
```

Then, run:
```bash
python3 aggregate_scores_mb.py \
        data/gg/ecbp/gold/test_events_corpus_level.conll \
        models/event_extraction/gg/ecbp/events_10_0.25/ecbp/closure
```

Afterwards, `models/event_extraction/gg/ecbp/events_10_0.25/ecbp/closure/scores.txt` will contain the scores (only the columns with `mentions` are relevant).

---

*Original project readme:*

# Cross-Document Coreference Resolution

This repository contains code and models for end-to-end cross-document coreference resolution, as decribed in our paper: [Streamlining Cross-Document Coreference Resolution: Evaluation and Modeling](https://arxiv.org/abs/2009.11032) 
The models are trained on ECB+, but they can be used for any setting of multiple documents.



```
    @article{Cattan2020StreamliningCC,
      title={Streamlining Cross-Document Coreference Resolution: Evaluation and Modeling},
      author={Arie Cattan and Alon Eirew and Gabriel Stanovsky and Mandar Joshi and I. Dagan},
      journal={ArXiv},
      year={2020},
      volume={abs/2009.11032}
    }
```


## Getting started

* Install python3 requirements `pip install -r requirements.txt` 
* Download spacy model `python -m spacy download en_core_web_sm`

### Extract mentions and raw text from ECB+ 

Run the following script in order to extract the data from ECB+ dataset
 and build the gold conll files. 
The ECB+ corpus can be downloaded [here](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).

* ``python get_ecb_data.py --data_path path_to_data``



## Training Instructions


The core of our model is the pairwise scorer between two spans, 
which indicates how likely two spans belong to the same cluster.


#### Training method

We present 3 ways to train this pairwise scorer:

1. Pipeline: first train a span scorer, then train the pairwise scorer. 
Unlike Ontonotes, ECB+ does include singleton annotation, so it's possible to train separately the span scorer model.
2.  Continue: first train the span scorer, then train the pairwise scorer
while continue training the span scorer.
3. End-to-end: train together the both models.

In order to choose the training method, you need to set the value of the `training_method` in 
the `config_pairwise.json` to `pipeline`, `continue` or `e2e`

In our experiments, we found the `e2e` method to perform the best for event coreference.

 
#### What are the labels ?

In ECB+, the entity and event coreference clusters are annotated separately, 
making it possible to train a model only on event or entity coreference. 
Therefore, our model also allows to be trained on events, entity, or both.
You need to set the value of the `mention_type` in 
the ``config_pairwise.json`` (and `config_span_scorer.json`) 
to `events`, `entities` or `mixed`.



#### Running the model
 
In both pipeline and fine-tuning methods, you need to first run 
the span scorer model 

* ``python train_span_scorer --config configs/config_span_scorer.json``

For the pairwise scorer, run the following script
* ``python train_pairwise_scorer configs/config_pairwise.json``



## Prediction

Given the pairwise scorer trained above, we use an agglomerative
clustering in order to cluster the candidate spans into coreference clusters. 


``python predict.py --config configs/config_clustering``

(`model_path` corresponds to the directory in which you've stored the trained models)

An important configuration in the `config_clustering` is the `topic_level`. 
If you set `false`, you need to provide the path to the predicted topics in `predicted_topics_path` 
to produce conll files at the corpus level. 

## Evaluation

The output of the `predict.py` script is a file in the standard conll format. 
Then, it's straightforward to evaluate it with its corresponding 
gold conll file (created in the first step), 
using the official conll coreference scorer
that you can find 
[here](https://github.com/conll/reference-coreference-scorers).

Make sure to use the gold files of the same evaluation level (topic or corpus) as the predictions. 


## Notes


* If you chose to train with the end-to-end method, you don't need to provide a `span_repr_path` or a `span_scorer_path` in the
config file.  

* Notice that if you use this model with gold mentions, 
the span scorer is not relevant, you should ignore the training
method.

* If you're interested in a newer model, check out our [cross-encoder model](https://github.com/ariecattan/cross_encoder/)