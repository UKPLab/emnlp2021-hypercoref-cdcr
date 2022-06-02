import collections
from pathlib import Path
from typing import Dict, Tuple

import click
import pyhocon
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from conll import write_output_file
from model_utils import *
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder
from train_pairwise_scorer_mb import _tweak_config_via_slurm, get_model_checkpoint_filenames
from train_pairwise_scorer_mb import cli_init, _generate_and_prune_mentions, CdcrDataset, \
    Collate, _predict_batch, regroup_mentions_by_doc_into_mentions_by_cluster
from utils import *


def is_included(docs, starts, ends, i1, i2):
    doc1, start1, end1 = docs[i1], starts[i1], ends[i1]
    doc2, start2, end2 = docs[i2], starts[i2], ends[i2]

    if doc1 == doc2 and (start1 >= start2 and end1 <= end2):
        return True
    return False


def remove_nested_mentions(cluster_ids, doc_ids, starts, ends):
    doc_ids = np.asarray(doc_ids)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    new_cluster_ids, new_docs_ids, new_starts, new_ends = [], [], [], []

    for cluster, idx in cluster_ids.items():
        docs = doc_ids[idx]
        start = starts[idx]
        end = ends[idx]

        for i in range(len(idx)):
            indicator = [is_included(docs, start, end, i, j) for j in range(len(idx))]
            if sum(indicator) > 1:
                continue

            new_cluster_ids.append(cluster)
            new_docs_ids.append(docs[i])
            new_starts.append(start[i])
            new_ends.append(end[i])

    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(new_cluster_ids):
        clusters[cluster_id].append(i)

    return clusters, new_docs_ids, new_starts, new_ends


def _do_predict(logger,
                config,
                device: torch.device,
                span_embedder_path: Path,
                span_scorer_path: Path,
                pairwise_scorer_path: Path,
                data_split: Corpus,
                thresholds: np.array,
                epoch: int,
                save_path: Path):
    logger.info(f"Check model from epoch {epoch}.")

    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    bert_model.eval()
    config['bert_hidden_size'] = bert_model.config.hidden_size

    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(str(span_embedder_path), map_location=device))
    span_repr.eval()

    span_scorer = SpanScorer(config).to(device)
    span_scorer.load_state_dict(torch.load(str(span_scorer_path), map_location=device))
    span_scorer.eval()

    pairwise_model = SimplePairWiseClassifier(config).to(device)
    pairwise_model.load_state_dict(torch.load(str(pairwise_scorer_path), map_location=device))
    pairwise_model.eval()

    use_gold_mentions = config["use_gold_mentions"]

    clusters_of_thresholds = [list() for _ in thresholds]
    max_ids = [0 for _ in thresholds]
    doc_ids, sentence_ids, starts, ends = [], [], [], []

    for topic in data_split.docs_by_topic.keys():
        logger.info(f"Processing topic {topic}")

        # determine spans to train on (these can be gold mentions or generated & pruned spans)
        if use_gold_mentions:
            mentions_with_cluster_id = data_split.mentions_with_cluster_id[
                topic]  # type: Dict[str, Dict[Tuple[int, int], int]]
        else:
            mentions_with_cluster_id = _generate_and_prune_mentions(data_split, topic)

        dataset = CdcrDataset(mentions_with_cluster_id, is_training=False)
        collate_fn = Collate(data_split, topic)
        loader = DataLoader(dataset, batch_size=config["batch_size_eval"], num_workers=1, collate_fn=collate_fn,
                            pin_memory=True)

        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(loader, unit="batch", mininterval=10):
                scores, labels = _predict_batch(batch,
                                                bert_model,
                                                span_repr,
                                                span_scorer,
                                                pairwise_model,
                                                device,
                                                include_span_scorer=not use_gold_mentions)
                scores = torch.sigmoid(scores)
                all_scores.extend(scores.detach().cpu())
                all_labels.append(labels.detach().cpu())

        all_scores = torch.stack(all_scores).numpy()

        # Affinity score to distance score
        pairwise_distances = 1 - squareform(all_scores)

        # collect info on original mentions
        clusters = regroup_mentions_by_doc_into_mentions_by_cluster(mentions_with_cluster_id)
        for _, mentions in clusters.items():
            for doc_id, (start, end) in mentions:
                doc_ids.append(doc_id)
                sentence_ids.append(data_split.documents[doc_id][start - 1][
                                        0])  # we heavily rely on token IDs being 1-based here (therefore the -1)!
                starts.append(start)
                ends.append(end)

        for i in range(len(thresholds)):
            agglomerative = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                    linkage=config['linkage_type'], distance_threshold=thresholds[i])
            predicted = agglomerative.fit(pairwise_distances)
            predicted_clusters = predicted.labels_ + max_ids[i]
            max_ids[i] = max(predicted_clusters) + 1
            clusters_of_thresholds[i].extend(predicted_clusters)

    for thresh, predicted in zip(thresholds, clusters_of_thresholds):
        logger.info('Saving clustering for threshold {}'.format(thresh))
        all_clusters = collections.defaultdict(list)
        for span_id, cluster_id in enumerate(predicted):
            all_clusters[cluster_id].append(span_id)

        if not use_gold_mentions:
            all_clusters, doc_ids, starts, ends = remove_nested_mentions(all_clusters, doc_ids, starts, ends)

        doc_name = 'model_{}_{}_{}_{}_{}'.format(epoch, config["split"], config['mention_type'], config['linkage_type'], thresh)

        write_output_file(data_split.documents, all_clusters, doc_ids, starts, ends, save_path, doc_name,
                          topic_level=config.topic_level, corpus_level=not config.topic_level, corpus_level_doc_name=f"dev_{config['mention_type']}")


def _predict_or_tune(config_path,
                     tune: bool = False):
    config = pyhocon.ConfigFactory.parse_file(config_path)
    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    _tweak_config_via_slurm(config, logger)

    if torch.cuda.device_count():
        gpu_to_use = config.gpu_num[0]
        device = torch.device(f"cuda:{gpu_to_use}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device {str(device)}")

    model_path = Path(config['model_path'])

    # run hyperopt / prediction over multiple seeds if available
    seed_dirs = [p for p in model_path.iterdir() if p.is_dir()] or [model_path]

    def iter_model_paths_tune(same_thresholds_for_all: np.array):
        for dir in seed_dirs:
            seed_save_path = Path(config["save_path"]) / dir.name
            seed_save_path.mkdir(parents=True, exist_ok=True)

            # iterate over checkpoints from all epochs available, starting with 0
            iter_epoch = 0
            while True:
                paths = [dir / fn for fn in get_model_checkpoint_filenames(iter_epoch)]
                if any(not p.exists() for p in paths):
                    break
                yield (iter_epoch, *paths, seed_save_path, same_thresholds_for_all)
                iter_epoch += 1

    def iter_model_paths_predict(best_hyperparams: Dict[str, Tuple[int, float]]):
        """
        :param best_hyperparams: per seed, the best epoch and clustering threshold
        """
        for dir in seed_dirs:
            # correct directories are those whose name can be interpreted as an integer (== seed)
            try:
                int(dir.name)
            except ValueError:
                continue

            seed_save_path = Path(config["save_path"]) / dir.name
            seed_save_path.mkdir(parents=True, exist_ok=True)

            epoch, threshold = best_hyperparams[dir.name]
            paths = [dir / fn for fn in get_model_checkpoint_filenames(epoch)]
            yield (epoch, *paths, seed_save_path, [threshold])

    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    if tune:
        config["split"] = "dev"
        data_split = Corpus.create_corpus(config, bert_tokenizer, config.split)
        same_thresholds_for_all = np.linspace(0.5, 0.7, num=5)
        model_paths_iterator = iter_model_paths_tune(same_thresholds_for_all)
    else:
        data_split = Corpus.create_corpus(config, bert_tokenizer, config.split, is_training=False)
        model_paths_iterator = iter_model_paths_predict(config["best_hyperparams"])

    for epoch, span_embedder_path, span_scorer_path, pairwise_scorer_path, save_path, thresholds in model_paths_iterator:
        _do_predict(logger,
                    config,
                    device,
                    span_embedder_path,
                    span_scorer_path,
                    pairwise_scorer_path,
                    data_split,
                    thresholds,
                    epoch,
                    save_path)


@cli_init.command(help="Tune clustering parameters")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def tune(config_path):
    _predict_or_tune(config_path, tune=True)

@cli_init.command(help="Predict (cluster) with a model")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def predict(config_path):
    _predict_or_tune(config_path, tune=False)


if __name__ == "__main__":
    cli_init()