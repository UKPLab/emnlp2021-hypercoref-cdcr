import collections
import json
import math
import sys
from itertools import combinations, repeat
from pathlib import Path
from random import sample, shuffle, choice, randint
from typing import List, Tuple, Dict, Optional, OrderedDict

import click
import pyhocon
from more_itertools import interleave_evenly
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from evaluator import Evaluation
from model_utils import *
from models import SpanEmbedder, SpanScorer, SimplePairWiseClassifier
from utils import *


@click.group()
@click.option("--debug", "debug", type=bool, default=False, help="Enable PyCharm remote debugger")
@click.option("--ip", "pycharm_debugger_ip", type=str, default=None, help="PyCharm debugger IP")
@click.option("--port", "pycharm_debugger_port", type=int, default=None, help="PyCharm debugger port")
def cli_init(debug, pycharm_debugger_ip, pycharm_debugger_port):
    # Before running any specific train/eval code, we start up the remote debugger if desired.
    if debug:
        try:
            # see https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html
            import pydevd_pycharm
            pydevd_pycharm.settrace(pycharm_debugger_ip, port=pycharm_debugger_port, stdoutToServer=True,
                                    stderrToServer=True)
        except ImportError as e:
            print("pydevd_pycharm is not installed. No remote debugging possible.")
            print(str(e))
            sys.exit(1)


def _tweak_config_via_slurm(config, logger):
    # overwrite batch size depending on node this script runs on - these work well for ECB+-sized topics
    node_name = os.getenv("SLURMD_NODENAME")
    gpus_in_nodes = {"some node": "p100", "some other node": "v100", "you get the idea": "a100"}
    batch_size_per_gpu_train = {"p100": 128, "v100": 128, "titanrtx": 128, "a100": 128}
    batch_size_per_gpu_eval = {"p100": 384, "v100": 1024, "titanrtx": 1024, "a100": 2048}
    if node_name in gpus_in_nodes:
        batch_size_train = batch_size_per_gpu_train[gpus_in_nodes[node_name]]
        batch_size_eval = batch_size_per_gpu_eval[gpus_in_nodes[node_name]]
    else:
        batch_size_train = batch_size_eval = 64
    config["batch_size"] = batch_size_train
    config["batch_size_train"] = batch_size_train
    config["batch_size_eval"] = batch_size_eval

    # if running in a job array, set random seed accordingly
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if array_task_id is not None:
        config["random_seed"] = int(array_task_id)
        config["model_path"] += f"/{array_task_id}"
        logger.info(
            f"Using random seed from Slurm array task id: {array_task_id}. Model is serialized to {config['model_path']}")


def _generate_and_prune_mentions(corpus: Corpus, topic: str) -> Dict[str, Dict[Tuple[int, int], int]]:
    """
    Generates all possible spans, embeds documents, scores spans, prunes bad spans and returns the good ones.
    """
    raise NotImplementedError


def regroup_mentions_by_doc_into_mentions_by_cluster(mentions_with_cluster_id) -> OrderedDict[int, List[Tuple[str, Tuple[int, int]]]]:
    # convert input data into usable format: dict mapping from cluster ID to mentions
    clusters = collections.OrderedDict()
    # (generated) spans without gold label will be assigned a dummy singleton cluster_id on the fly
    next_dummy_cluster_id = max(cluster_id for d in mentions_with_cluster_id.values() for cluster_id in d.values()) + 1
    for doc_id, mentions in mentions_with_cluster_id.items():
        for span, cluster_id in mentions.items():
            # use dummy singleton id if no gold label is present
            if cluster_id is None:
                cluster_id = next_dummy_cluster_id
                next_dummy_cluster_id += 1
            list_of_mentions_for_cluster = clusters.get(cluster_id, [])
            list_of_mentions_for_cluster.append((doc_id, span))
            clusters[cluster_id] = list_of_mentions_for_cluster
    return clusters


class CdcrDataset(IterableDataset):
    """
    Because n (total number of mentions) can be very large with certain datasets, we don't want to ever generate all
    possible mention pairs in memory. We use iterators (and IterableDataset) instead. For training, we sample all
    positive pairs based on clusters, then sample a fixed amount of random negative pairs. Sufficient randomness in the
    output order of pairs is ensured by spreading positive/negative pairs with Bresenham, followed by a buffer which
    collects and shuffles instances.
    For testing, we yield all mention pair combinations.
    See https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd for instructions
    on how to parallelize IterableDataset properly in the future (if necessary).
    """
    def __init__(self,
                 mentions_with_cluster_id,
                 neg_pair_factor: int = 15,
                 is_training: bool = False):
        self.is_training = is_training
        self.clusters = regroup_mentions_by_doc_into_mentions_by_cluster(mentions_with_cluster_id)

        # ----- determine number of pairs ------
        # positive pairs
        self.num_pos_pairs = 0
        for mentions in self.clusters.values():
            n = len(mentions)
            if n < 2:
                continue

            # for large n, using all pairs during training (1) is not very useful as most pairs are similar, and (2) it
            # slows down training
            num_pos_pairs = (n * (n - 1)) // 2
            if is_training:
                # simple cap for number of mentions
                num_pos_pairs = min(num_pos_pairs, int(6 * math.sqrt(n)))
            self.num_pos_pairs += num_pos_pairs

        # negative pairs
        num_mentions = sum(map(len,self.clusters.values()))
        num_total_pairs = (num_mentions * (num_mentions - 1)) // 2
        self.num_neg_pairs = num_total_pairs - self.num_pos_pairs
        if self.is_training:
            self.num_neg_pairs = min(neg_pair_factor * self.num_pos_pairs, self.num_neg_pairs)

    def _generate_pairs_training(self):
        # TODO maybe split effort across multiple workers here, using worker_init_fn, and use random seed from worker_info

        def yield_pos_pairs(clusters):
            # since we undersample pos pairs, shuffle cluster so that training data consists of different clusters
            # every epoch
            cluster_ids = list(clusters.keys())
            shuffle(cluster_ids)
            for cluster_id in cluster_ids:
                mentions = clusters[cluster_id]
                if len(mentions) < 2:
                    continue
                for pair in combinations(mentions, 2):
                    yield pair

        def yield_neg_pairs(clusters, num_pairs: int):
            set_of_cluster_ids = set(clusters.keys())
            for _ in range(num_pairs):
                # sample two random clusters, then sample one mention from each cluster
                cluster_a, cluster_b = sample(set_of_cluster_ids, 2)
                mention_a = choice(clusters[cluster_a])
                mention_b = choice(clusters[cluster_b])
                yield mention_a, mention_b

        # add labels to each instance
        pos_pairs_with_label = zip(yield_pos_pairs(self.clusters), repeat(True, self.num_pos_pairs))
        neg_pairs_with_label = zip(yield_neg_pairs(self.clusters, self.num_neg_pairs), repeat(False, self.num_neg_pairs))

        # interleave positive and negative pairs evenly
        instances = interleave_evenly([pos_pairs_with_label, neg_pairs_with_label],
                                      lengths=[self.num_pos_pairs, self.num_neg_pairs])
        return instances

    def _generate_pairs_testing(self):
        """
        Generate all n*(n-1)/2 pairs in sequence.
        """
        def mentions_with_cluster_id(clusters):
            for cluster_id, mentions in clusters.items():
                for mention in mentions:
                    yield mention, cluster_id

        for (mention_a, id_a), (mention_b, id_b) in combinations(mentions_with_cluster_id(self.clusters), 2):
            yield (mention_a, mention_b), id_a == id_b

    def __iter__(self):
        iterable = self._generate_pairs_training() if self.is_training else self._generate_pairs_testing()
        yield from iterable

    def __len__(self):
        return self.num_neg_pairs + self.num_pos_pairs


class Collate:
    """
    Looks up correct BERT wordpiece tokens for mention pairs. Assembles batches.
    """

    def __init__(self,
                 corpus: Corpus,
                 topic: str):
        self.corpus = corpus
        self.topic = topic

    def __call__(self, batch):
        # We want to minimize the number of BERT model invocations in each batch. Strategy: we identify which document
        # segments are required for the mentions in a batch, put them in a list and remember for each mention which
        # segment (== index into this list) they belong to. We then retrieve BERT tokens only for the required segments,
        # embed only those segments and use the remembered index per mention to look up the correct embeddings for each.
        segments_of_batch = []
        pre_collate_instances = []

        mentions_to_segments = self.corpus.mentions_to_segment_ids[self.topic]
        for (mention_a, mention_b), label in batch:
            pre_collate_instance = []
            for doc_id, (span_origin_start_token_id, span_origin_end_token_id) in [mention_a, mention_b]:
                segment = mentions_to_segments[doc_id][(span_origin_start_token_id, span_origin_end_token_id)]

                # if segment was used for this batch already, look up its index, otherwise add new
                if segment in segments_of_batch:
                    segment_idx_in_batch = segments_of_batch.index(segment)
                else:
                    segment_idx_in_batch = len(segments_of_batch)
                    segments_of_batch.append(segment)

                # find the indices of the span's original tokens in the segment
                segment_original_tokens_ids = [x[1] for x in self.corpus.origin_tokens[self.topic][segment]]
                span_origin_start_idx = segment_original_tokens_ids.index(span_origin_start_token_id)
                span_origin_end_idx = segment_original_tokens_ids.index(span_origin_end_token_id)
                # then look up the indices of the BERT wordpieces corresponding to the span's start/end
                span_bert_start_idx = self.corpus.start_end_bert[self.topic][segment][span_origin_start_idx][0]
                span_bert_end_idx = self.corpus.start_end_bert[self.topic][segment][span_origin_end_idx][1]
                span_origin_width = span_origin_end_idx - span_origin_start_idx
                pre_collate_instance += [segment_idx_in_batch, span_bert_start_idx, span_bert_end_idx, span_origin_width]

            pre_collate_instance.append(label)
            assert len(pre_collate_instance) == 9

            pre_collate_instances.append(pre_collate_instance)

        instances = default_collate(pre_collate_instances)

        # now retrieve BERT tokens (pad jagged array)
        tokens_tensors = [torch.LongTensor(self.corpus.bert_tokens[self.topic][idx]) for idx in segments_of_batch]
        tokens_tensors_padded = torch.nn.utils.rnn.pad_sequence(tokens_tensors, batch_first=True)
        tokens_tensors_mask = (tokens_tensors_padded != 0).to(dtype=torch.long)

        return tokens_tensors_padded, tokens_tensors_mask, instances


class ShuffleDataset(IterableDataset):
    """
    Copied 1:1 from https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
    """
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def _predict_batch(batch: Tuple,
                   bert_model,
                   span_repr,
                   span_scorer,
                   pairwise_model,
                   device: torch.device,
                   include_span_scorer: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Embed spans and train classifier and/or span scorer. Returns loss.
    """
    assert len(batch) == 3
    tokens_padded, tokens_mask, instances = batch
    tokens_padded = tokens_padded.to(device, non_blocking=True)
    tokens_mask = tokens_mask.to(device, non_blocking=True)
    a_segments, a_starts, a_ends, a_widths, b_segments, b_starts, b_ends, b_widths, labels = map(lambda t: t.to(device, non_blocking=True), instances)

    # sanity checks
    num_instances = a_segments.shape[0]
    assert all(t.shape[0] == num_instances for t in [a_starts, a_ends, a_widths, b_segments, b_starts, b_ends, b_widths, labels])

    # BERT embed documents
    with torch.no_grad():
        embedded, _ = bert_model(tokens_padded, tokens_mask)

    # Looking up the correct segment embedding for each mention with index_select allocates new memory. As the size
    # of that tensor is known in advance, we reuse the same output vector for these operations to keep memory usage
    # constant.
    tmp_embedded_selection = embedded.new_empty((num_instances, embedded.shape[1], embedded.shape[2]))

    # make "start_end_embeddings" and "continuous_embeddings" for a and b
    a_or_b_start_end_embeddings = []
    a_or_b_continuous_embeddings = []
    for segments, (start, end) in zip([a_segments, b_segments], [(a_starts, a_ends), (b_starts, b_ends)]):
        embedded_segments = torch.index_select(embedded, 0, segments, out=tmp_embedded_selection)
        spans = torch.hstack([start.view(-1, 1), end.view(-1, 1)])   # shape: (num_instances, 2) <- start, end for each instance

        # take embedding at start and end, then flatten
        start_end_embeddings = []
        for i, span in enumerate(spans):
            start_end_embedding = embedded_segments[i, span, :].view(-1)     # shape: (2*1024)
            start_end_embeddings.append(start_end_embedding)
        start_end_embeddings = torch.vstack(start_end_embeddings)   # shape: (num_instances, 2*1024)
        a_or_b_start_end_embeddings.append(start_end_embeddings)

        # make span ends exclusive for use with torch.arange, then look up ranges
        spans_excl_end = spans.clone().detach()
        spans_excl_end[:, 1] += 1
        continuous_embeddings = []
        for i, span in enumerate(spans_excl_end):
            continuous_span_indices = torch.arange(*span, device=span.device)
            continuous_embedding = embedded_segments[i, continuous_span_indices, :]      # shape: (-1, 1024)
            continuous_embeddings.append(continuous_embedding)
        a_or_b_continuous_embeddings.append(continuous_embeddings)
    a_start_end_embeddings, b_start_end_embeddings = a_or_b_start_end_embeddings
    a_continuous_embeddings, b_continuous_embeddings = a_or_b_continuous_embeddings

    # compute span representations
    a = span_repr(a_start_end_embeddings,
                  a_continuous_embeddings,
                  a_widths)
    b = span_repr(b_start_end_embeddings,
                  b_continuous_embeddings,
                  b_widths)
    scores = pairwise_model(a, b)

    if include_span_scorer:
        a_score = span_scorer(a)
        b_score = span_scorer(b)
        scores += a_score + b_score

    scores = scores.squeeze(1)

    # free memory
    del tmp_embedded_selection
    return scores, labels


def get_summary_path(config_path) -> Path:
    root_path = Path("runs")
    job_id = os.environ.get("SLURM_JOB_ID", "non-slurm")
    config_path_str = str(config_path).replace("/", "_")
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{job_id}_{config_path_str}_{date_str}"
    return root_path / name


get_model_checkpoint_filenames = lambda e: [f"{prefix}_{e}" for prefix in ["span_repr", "span_scorer", "pairwise_scorer"]]


def get_last_valid_model_checkpoints(model_path: Path) -> Tuple[int, Optional[List[Path]], List[float]]:
    dev_losses_file = model_path / "dev_losses.json"
    if not dev_losses_file.exists():
        return -1, None, []

    # determine last epoch for which training and validation completed
    with dev_losses_file.open() as f:
        dev_losses = json.load(f)
    last_valid_epoch = len(dev_losses) - 1

    # find corresponding model checkpoints
    paths = [model_path / fn for fn in get_model_checkpoint_filenames(last_valid_epoch)]
    if not all(p.exists() for p in paths):
        raise ValueError(f"Model checkpoints not found for epoch {last_valid_epoch}")

    return last_valid_epoch, paths, dev_losses


@cli_init.command(help="Train a model")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def train(config_path):
    config = pyhocon.ConfigFactory.parse_file(config_path)
    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    _tweak_config_via_slurm(config, logger)

    fix_seed(config)
    model_path = Path(config["model_path"])
    model_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.device_count():
        gpu_to_use = config.gpu_num[0]
        device = torch.device(f"cuda:{gpu_to_use}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device {str(device)}")

    # init train and dev set
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    training_set = Corpus.create_corpus(config, bert_tokenizer, 'train')
    dev_set = Corpus.create_corpus(config, bert_tokenizer, 'dev')
    num_training_topics = len(training_set.docs_by_topic)
    num_dev_topics = len(dev_set.docs_by_topic)

    ## Model initiation
    logger.info('Init models')
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    bert_model.eval()
    config['bert_hidden_size'] = bert_model.config.hidden_size

    span_repr = SpanEmbedder(config, device).to(device)
    span_scorer = SpanScorer(config).to(device)
    pairwise_model = SimplePairWiseClassifier(config).to(device)

    use_gold_mentions = config["use_gold_mentions"]
    training_method = config["training_method"]
    resume_training_if_possible = config.get("resume_training_if_possible", True)

    # set up optimizer and loss function
    models = [pairwise_model]
    if not use_gold_mentions:
        if training_method in ["pipeline", "continue"]:
            span_repr.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
            span_scorer.load_state_dict(torch.load(config['span_scorer_path'], map_location=device))
        if training_method in ["e2e", "continue"]:
            models.append(span_repr)
            models.append(span_scorer)

    optimizer = get_optimizer(config, models)
    criterion = get_loss_function(config)

    logger.info('Number of parameters of mention extractor: {}'.format(count_parameters(span_repr) + count_parameters(span_scorer)))
    logger.info('Number of parameters of the pairwise classifier: {}'.format(count_parameters(pairwise_model)))
    logger.info('Number of topics: {}'.format(len(training_set.docs_by_topic)))

    epochs = range(config["epochs"])
    dev_losses = []
    if resume_training_if_possible:
        last_valid_epoch, checkpoint_paths, dev_losses = get_last_valid_model_checkpoints(model_path)
        if last_valid_epoch == -1:
            logger.info(f"No model checkpoints to resume training from, starting fresh...")
        else:
            logger.info(f"Resuming training with checkpoints from epoch {last_valid_epoch}.")
            span_repr_checkpoint, span_scorer_checkpoint, pairwise_model_checkpoint = checkpoint_paths
            span_repr.load_state_dict(torch.load(span_repr_checkpoint, map_location=device))
            span_scorer.load_state_dict(torch.load(span_scorer_checkpoint, map_location=device))
            pairwise_model.load_state_dict(torch.load(pairwise_model_checkpoint, map_location=device))
            epochs = range(last_valid_epoch +1, config["epochs"])

    patience = config.get("patience", 7)
    summary_writer = SummaryWriter(log_dir=str(get_summary_path(config_path)))
    for epoch in epochs:
        # check early stopping
        if len(dev_losses) > patience:
            min_dev_loss = min(dev_losses)
            if all(l > min_dev_loss for l in dev_losses[-patience:]):
                best_epoch = np.argmin(np.array(dev_losses))
                logger.info(f"The last {patience} epochs didn't improve over the best dev loss of {min_dev_loss} from epoch {best_epoch}. Early stopping.")
                break

        logger.info(f"Epoch: {epoch}")

        pairwise_model.train()
        if training_method in ["continue", "e2e"] and not use_gold_mentions:
            span_repr.train()
            span_scorer.train()
        else:
            span_repr.eval()
            span_scorer.eval()

        train_loss = 0
        list_of_topics = sample(list(training_set.docs_by_topic), k=len(training_set.docs_by_topic))
        for topic in list_of_topics:
            logger.info(f"Training on topic {topic}")

            # determine spans to train on (these can be gold mentions or generated & pruned spans)
            if use_gold_mentions:
                mentions_with_cluster_id = training_set.mentions_with_cluster_id[topic]     # type: Dict[str, Dict[Tuple[int, int], int]]
            else:
                mentions_with_cluster_id = _generate_and_prune_mentions(training_set, topic)

            dataset = CdcrDataset(mentions_with_cluster_id, is_training=config.get("neg_samp", True), neg_pair_factor=config.get("neg_samp_factor", 5))
            dataset = ShuffleDataset(dataset, buffer_size=2048)
            collate_fn = Collate(training_set, topic)
            loader = DataLoader(dataset, batch_size=config["batch_size_train"], num_workers=1, collate_fn=collate_fn, pin_memory=True)

            train_loss_topic = 0
            for batch in tqdm(loader, unit="batch", mininterval=10):
                optimizer.zero_grad()
                scores, labels = _predict_batch(batch,
                                                bert_model,
                                                span_repr,
                                                span_scorer,
                                                pairwise_model,
                                                device,
                                                include_span_scorer=training_method in ["continue", "e2e"] and not use_gold_mentions)
                # -1 negative label when using hinge loss
                if config["loss"] == "hinge":
                    labels_f = torch.where(labels, 1.0, -1.0)
                else:
                    labels_f = labels.to(dtype=torch.float)

                loss = criterion(scores, labels_f)
                loss.backward()
                optimizer.step()
                train_loss_topic += loss.detach().item()
            # log the loss per topic, but not if there are too many (tensorboard summaries get too large otherwise)
            if num_training_topics < 50:
                summary_writer.add_scalar(f"loss/train/topic/{topic}", train_loss_topic, epoch)
            train_loss += train_loss_topic
        summary_writer.add_scalar(f"loss/train", train_loss, epoch)
        logger.info(f"Train loss: {train_loss}")

        logger.info('Evaluating on the dev set')
        span_repr.eval()
        span_scorer.eval()
        pairwise_model.eval()

        dev_loss = 0
        all_predictions, all_labels = [], []
        for topic in dev_set.docs_by_topic.keys():
            logger.info(f"Evaluating topic {topic}")

            # determine spans to train on (these can be gold mentions or generated & pruned spans)
            if use_gold_mentions:
                mentions_with_cluster_id = dev_set.mentions_with_cluster_id[topic]     # type: Dict[str, Dict[Tuple[int, int], int]]
            else:
                mentions_with_cluster_id = _generate_and_prune_mentions(dev_set, topic)

            dataset = CdcrDataset(mentions_with_cluster_id, is_training=False)
            collate_fn = Collate(dev_set, topic)
            loader = DataLoader(dataset, batch_size=config["batch_size_eval"], num_workers=1, collate_fn=collate_fn, pin_memory=True)

            with torch.no_grad():
                dev_loss_topic = 0
                for batch in tqdm(loader, unit="batch", mininterval=10):
                    scores, labels = _predict_batch(batch,
                                                    bert_model,
                                                    span_repr,
                                                    span_scorer,
                                                    pairwise_model,
                                                    device,
                                                    include_span_scorer=training_method in ["continue", "e2e"] and not use_gold_mentions)
                    # -1 negative label when using hinge loss
                    if config["loss"] == "hinge":
                        labels_f = torch.where(labels, 1.0, -1.0)
                    else:
                        labels_f = labels.to(dtype=torch.float)

                    loss = criterion(scores, labels_f)
                    dev_loss_topic += loss.detach().item()
                    all_predictions.append((scores > 0.0).detach().cpu())
                    all_labels.append(labels.detach().cpu())
            # log the loss per topic, but not if there are too many (tensorboard summaries get too large otherwise)
            if num_dev_topics < 50:
                summary_writer.add_scalar(f"loss/dev/topic/{topic}", dev_loss_topic, epoch)
            dev_loss += dev_loss_topic

        # check R, P, F1
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        eval = Evaluation(all_predictions.to(device), all_labels.to(device))
        r, p, f1 = eval.get_recall(), eval.get_precision(), eval.get_f1()
        logger.info(f'Strict - Recall: {r}, Precision: {p}, F1: {f1}')
        logger.info(f"Dev loss: {dev_loss}")
        summary_writer.add_scalar("recall/dev", r, epoch)
        summary_writer.add_scalar("precision/dev", p, epoch)
        summary_writer.add_scalar("f1/dev", f1, epoch)
        dev_losses.append(dev_loss)
        summary_writer.add_scalar(f"loss/dev", dev_loss, epoch)

        torch.save(span_repr.state_dict(), model_path / f"span_repr_{epoch}")
        torch.save(span_scorer.state_dict(), model_path / f"span_scorer_{epoch}")
        torch.save(pairwise_model.state_dict(), model_path / f"pairwise_scorer_{epoch}")
        with (model_path / "dev_losses.json").open("w") as f:
            json.dump(dev_losses, f)

    summary_writer.close()
    logger.info("Complete.")

if __name__ == '__main__':
    cli_init()