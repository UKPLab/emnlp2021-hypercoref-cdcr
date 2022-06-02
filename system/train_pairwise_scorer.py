import sys
import json
from itertools import combinations
from pathlib import Path

import click
import pyhocon
from sklearn.utils import shuffle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from evaluator import Evaluation
from model_utils import *
from models import SpanEmbedder, SpanScorer, SimplePairWiseClassifier
from spans import TopicSpans
from train_pairwise_scorer_mb import _tweak_config_via_slurm, get_last_valid_model_checkpoints, \
    get_model_checkpoint_filenames
from utils import *

import warnings

warnings.warn("Deprecated in favor of `train_pairwise_scorer_mb.py`, except for experiments on ECB+. This version loads entire topics into GPU memory which fails for other corpora.", DeprecationWarning)


def train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings,
                                    first, second, labels, batch_size, criterion, optimizer):
    accumulate_loss = 0
    start_end_embeddings, continuous_embeddings, width = span_embeddings
    device = start_end_embeddings.device
    labels = labels.to(device)
    # width = width.to(device)

    idx = shuffle(list(range(len(first))))
    for i in range(0, len(first), batch_size):
        indices = idx[i:i+batch_size]
        batch_first, batch_second = first[indices], second[indices]
        batch_labels = labels[indices].to(torch.float)
        optimizer.zero_grad()
        g1 = span_repr(start_end_embeddings[batch_first],
                                [continuous_embeddings[k] for k in batch_first], width[batch_first])
        g2 = span_repr(start_end_embeddings[batch_second],
                                [continuous_embeddings[k] for k in batch_second], width[batch_second])
        scores = pairwise_model(g1, g2)

        if config['training_method'] in ('continue', 'e2e') and not config["use_gold_mentions"]:
            g1_score = span_scorer(g1)
            g2_score = span_scorer(g2)
            scores += g1_score + g2_score

        loss = criterion(scores.squeeze(1), batch_labels)
        accumulate_loss += loss.item()
        loss.backward()
        optimizer.step()

    return accumulate_loss


def get_all_candidate_spans(config, bert_model, span_repr, span_scorer, data, topic: str, batch_size: int, use_gold_mentions: bool):
    docs_embeddings, docs_length = pad_and_read_bert(data.bert_tokens[topic], bert_model)
    topic_spans = TopicSpans(config, data, topic, docs_embeddings, docs_length, is_training=True)

    topic_spans.set_span_labels(topic)

    ## Pruning the spans according to gold mentions or spans with highiest scores
    if use_gold_mentions:
        span_indices = topic_spans.labels.nonzero().squeeze(1)
    else:
        k = int(config['top_k'] * topic_spans.num_tokens)
        with torch.no_grad():
            span_scores = []
            slightly_larger_batch_size = 10 * batch_size
            for i in range(0, len(topic_spans.continuous_embeddings), slightly_larger_batch_size):
                starts_ends = topic_spans.start_end_embeddings[i:i+slightly_larger_batch_size]
                continuouses = topic_spans.continuous_embeddings[i:i + slightly_larger_batch_size]
                widths = topic_spans.width[i:i+slightly_larger_batch_size]
                span_emb = span_repr(starts_ends,
                                     continuouses,
                                     widths)
                batch_span_scores = span_scorer(span_emb)
                span_scores.append(batch_span_scores.detach().cpu())
            span_scores = torch.cat(span_scores)
            _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)

    topic_spans.prune_spans(span_indices)
    torch.cuda.empty_cache()

    return topic_spans


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


def get_pairwise_labels(labels, device, is_training: bool = False, use_hinge_loss: bool = False):
    first, second = zip(*list(combinations(range(len(labels)), 2)))
    first = torch.tensor(first)
    second = torch.tensor(second)
    pairwise_labels = (labels[first] != 0) & (labels[second] != 0) & \
                      (labels[first] == labels[second])

    if is_training:
        positives = (pairwise_labels == 1).nonzero().view([-1])
        positive_ratio = len(positives) / len(first)
        negatives = (pairwise_labels != 1).nonzero().view([-1])
        rands = torch.rand(len(negatives))
        rands = (rands < positive_ratio * 20).to(torch.long)
        sampled_negatives = negatives[rands.nonzero().squeeze()]
        new_first = torch.cat((first[positives], first[sampled_negatives]))
        new_second = torch.cat((second[positives], second[sampled_negatives]))
        new_labels = torch.cat((pairwise_labels[positives], pairwise_labels[sampled_negatives]))
        first, second, pairwise_labels = new_first, new_second, new_labels

    pairwise_labels = pairwise_labels.to(torch.long).to(device)

    if use_hinge_loss:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device))
    else:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))
    torch.cuda.empty_cache()

    return first, second, pairwise_labels


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

    ## Model initiation
    logger.info('Init models')
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size

    span_repr = SpanEmbedder(config, device).to(device)
    span_scorer = SpanScorer(config).to(device)
    pairwise_model = SimplePairWiseClassifier(config).to(device)

    use_gold_mentions = config["use_gold_mentions"]
    training_method = config["training_method"]
    resume_training_if_possible = config.get("resume_training_if_possible", True)

    # set up optimizer and loss function
    models = [pairwise_model]
    if training_method in ('pipeline', 'continue') and not use_gold_mentions:
        span_repr.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
        span_scorer.load_state_dict(torch.load(config['span_scorer_path'], map_location=device))
    if training_method in ('continue', 'e2e') and not use_gold_mentions:
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

    # early stopping
    patience = config.get("patience", 7)
    for epoch in epochs:
        # check early stopping
        if len(dev_losses) > patience:
            min_dev_loss = min(dev_losses)
            if all(l > min_dev_loss for l in dev_losses[-patience:]):
                best_epoch = np.argmin(np.array(dev_losses))
                logger.info(f"The last {patience} epochs didn't improve over the best dev loss of {min_dev_loss} from epoch {best_epoch}. Early stopping.")
                break

        logger.info('Epoch: {}'.format(epoch))

        pairwise_model.train()
        if training_method in ('continue', 'e2e') and not use_gold_mentions:
            span_repr.train()
            span_scorer.train()
        else:
            span_repr.eval()
            span_scorer.eval()

        accumulate_loss = 0

        list_of_topics = shuffle(list(training_set.docs_by_topic.keys()))
        total_number_of_pairs = 0
        for topic in tqdm(list_of_topics):
            topic_spans = get_all_candidate_spans(config, bert_model, span_repr, span_scorer, training_set, topic, config["batch_size"], use_gold_mentions=use_gold_mentions)
            first, second, pairwise_labels = get_pairwise_labels(topic_spans.labels, device, is_training=config['neg_samp'], use_hinge_loss=config['loss'] == 'hinge')
            span_embeddings = topic_spans.start_end_embeddings, topic_spans.continuous_embeddings, topic_spans.width
            loss = train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings, first,
                                                   second, pairwise_labels, config['batch_size'], criterion, optimizer)
            torch.cuda.empty_cache()
            accumulate_loss += loss
            total_number_of_pairs += len(first)

        logger.info('Number of training pairs: {}'.format(total_number_of_pairs))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))


        logger.info('Evaluate on the dev set')

        span_repr.eval()
        span_scorer.eval()
        pairwise_model.eval()

        all_scores, all_labels = [], []
        dev_loss = 0
        for topic in tqdm(dev_set.docs_by_topic.keys()):
            topic_spans = get_all_candidate_spans(config, bert_model, span_repr, span_scorer, dev_set, topic, config["batch_size"], use_gold_mentions=True)
            first, second, pairwise_labels = get_pairwise_labels(topic_spans.labels, device, is_training=False, use_hinge_loss=config['loss'] == 'hinge')

            topic_spans.width = topic_spans.width.to(device)
            with torch.no_grad():
                dev_loss_topic = 0
                for i in range(0, len(first), 10000):
                    end_max = i + 10000
                    first_idx, second_idx = first[i:end_max], second[i:end_max]
                    batch_labels = pairwise_labels[i:end_max]
                    g1 = span_repr(topic_spans.start_end_embeddings[first_idx],
                                   [topic_spans.continuous_embeddings[k] for k in first_idx],
                                   topic_spans.width[first_idx])
                    g2 = span_repr(topic_spans.start_end_embeddings[second_idx],
                                   [topic_spans.continuous_embeddings[k] for k in second_idx],
                                   topic_spans.width[second_idx])
                    scores = pairwise_model(g1, g2)

                    if training_method in ('continue', 'e2e') and not use_gold_mentions:
                        g1_score = span_scorer(g1)
                        g2_score = span_scorer(g2)
                        scores += g1_score + g2_score

                    loss = criterion(scores.squeeze(1), batch_labels.to(torch.float))
                    dev_loss_topic += loss.detach().item()
                    all_scores.extend(scores.squeeze(1))
                    all_labels.extend(batch_labels.to(torch.int))
                dev_loss += dev_loss_topic

        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)
        logger.info(f"Dev loss: {dev_loss}")
        dev_losses.append(dev_loss)

        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                             len(all_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                        eval.get_precision(), eval.get_f1()))

        torch.save(span_repr.state_dict(), model_path / f"span_repr_{epoch}")
        torch.save(span_scorer.state_dict(), model_path / f"span_scorer_{epoch}")
        torch.save(pairwise_model.state_dict(), model_path / f"pairwise_scorer_{epoch}")
        with (model_path / "dev_losses.json").open("w") as f:
            json.dump(dev_losses, f)



if __name__ == '__main__':
    cli_init()