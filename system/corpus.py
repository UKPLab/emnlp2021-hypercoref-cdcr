import collections
import json
import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import tqdm


class Corpus:
    def __init__(self,
                 documents: Dict[str, List[Tuple[int, int, str, bool]]],
                 mentions: List[Dict],
                 subtopic=True,
                 predicted_topics=None):
        """
        :param documents: each document is a list of (sent_id, token_id, token, is_validated) tuples
        """
        self.documents = documents
        if predicted_topics:
            self.docs_by_topic = self.external_document_partitioning(predicted_topics)
        else:
            self.docs_by_topic = self.gold_document_partitioning(by_subtopic=subtopic)

        # per topic: the list of original doc_ids corresponding to each segment
        self.list_of_docs = None    # type: Optional[Dict[str, List[str]]]

        # per topic: the list of original tokens corresponding to each segment
        self.origin_tokens = None   # type: Optional[Dict[str, List[List[Tuple[int, int, str, bool]]]]]

        # per topic: the list of BERT wordpiece tokens corresponding to each segment
        self.bert_tokens = None     # type: Optional[Dict[str, List[List[int]]]]

        # per topic: the start/end indices of each token in the wordpieces of a segment
        self.start_end_bert = None  # type: Optional[Dict[str, List[List[np.array]]]]

        # per topic, all documents and their mentions plus cluster ID as a dict
        self.mentions_with_cluster_id = {}  # type: Dict[str, Dict[str, Dict[Tuple[int, int], int]]]
        mentions_with_cluster_id_by_document = collections.defaultdict(dict)
        for m in mentions:
            span = m["tokens_ids"]
            doc_id = m["doc_id"]
            span_start_end = (min(span), max(span))

            # check if this mention belongs to one of the validated sentences - if it doesn't, scrap it
            keep = False
            sent_id = int(m["sentence_id"])
            for tok_sent_id, _, _, is_validated in documents[doc_id]:
                if tok_sent_id == sent_id and is_validated:
                    keep = True
                    break
            if not keep:
                continue

            mentions_with_cluster_id_by_document[doc_id][span_start_end] = m["cluster_id"]
        for topic, doc_ids in self.docs_by_topic.items():
            self.mentions_with_cluster_id[topic] = {doc_id: mentions_with_cluster_id_by_document[doc_id] for doc_id in doc_ids}

        # mapping from mentions to the index of their BERT_tokenized surrounding segment of their span
        self.mentions_to_segment_ids = None     # type: Optional[Dict[str, Dict[str, Dict[Tuple[int, int], int]]]]

    def get_candidate_labels(self,
                             topic: str,
                             doc_ids: List[str],
                             starts: torch.LongTensor,
                             ends: torch.LongTensor):
        mentions_with_cluster_id_of_topic = self.mentions_with_cluster_id[topic]
        assert all(doc_id in mentions_with_cluster_id_of_topic.keys() for doc_id in doc_ids), f"Expected {str(doc_ids)} to be in topic {topic}, but not all of them were."

        starts_list = starts.tolist()
        ends_list = ends.tolist()

        labels = [mentions_with_cluster_id_of_topic[doc_id].get((start, end), 0) for doc_id, start, end in zip(doc_ids, starts_list, ends_list)]
        return torch.tensor(labels)

    def external_document_partitioning(self, predicted_subtopics) -> Dict[str, List[str]]:
        '''
        Function to init the predicted subtopics as Shany Barhom
        :param predicted_subtopics: Shany's file
        :return:
        '''
        docs_by_topics_external = collections.defaultdict(list)
        for i, doc_list in enumerate(predicted_subtopics):
            for doc in doc_list:
                if doc not in self.documents:
                    doc += ".xml"   # fix for Arie's format where document names include the file extension
                assert doc in self.documents
                docs_by_topics_external[f"external-topic-{i}"].append(doc)
        return docs_by_topics_external

    def gold_document_partitioning(self, by_subtopic: bool) -> Dict[str, List[str]]:
        """
        :param by_subtopic: if True, partition by gold subtopic, otherwise by gold topic
        """
        docs_by_topics = collections.defaultdict(list)
        for doc_id, tokens in self.documents.items():
            doc_id_parts = doc_id.split("_")
            topic_key = doc_id_parts[0]
            if by_subtopic:
                topic_key += "_" + doc_id_parts[1]
            docs_by_topics[topic_key].append(doc_id)

        return docs_by_topics

    @staticmethod
    def _split_doc_into_segments(token_alignment,
                                 sentence_alignment,
                                 max_segment_length: int) -> List[int]:
        """
        Determine token indices at which a document needs to be split into segments so that each segment fits into the
        transformer input while keeping document sentences intact.
        """
        segments = [0]
        segment_start = 0

        while segment_start < len(token_alignment):
            num_remaining_tokens = len(token_alignment) - segment_start
            if num_remaining_tokens > max_segment_length:
                # try to fit segments to sentences: if the token at index `segment_end` (which is just outside
                # the current segment) has the same sentence ID, move segment end back by 1 token, and so on
                segment_end = segment_start + max_segment_length
                while sentence_alignment[segment_end - 1] == sentence_alignment[segment_end] and segment_end > segment_start:
                    segment_end -= 1

                # If at this point, the segment end candidate is equal to the segment start, the sentence has more
                # tokens than the transformer input sequence can fit, so we unfortunately need to chop the sentence up.
                if segment_end == segment_start:
                    segment_end = segment_start + max_segment_length

                    # similarly to sentences above, use token alignment to cut at the next best token
                    while token_alignment[segment_end -1] == token_alignment[segment_end] and segment_end > segment_start:
                        segment_end -= 1

                    if segment_end == segment_start:
                        raise ValueError(f"Transformer input sequence length is too short to fit in the token at index {segment_start}")
            else:
                segment_end = len(token_alignment)

            segments.append(segment_end)
            segment_start = segment_end
        return segments

    @staticmethod
    def _bert_tokenize_document(doc_id: str,
                                tokens: List[Tuple[int, int, str, bool]],
                                tokenizer,
                                segment_window: int) -> Tuple[List[np.array], List[str], List[List[int]], List[List[Tuple[int, int, str, bool]]]]:
        bert_tokens_ids, sentence_alignment = [], []
        start_bert_idx, end_bert_idx = [], []
        token_alignment = []
        bert_cursor = 0

        for i, token in enumerate(tokens):
            sent_id, token_id, token_text, flag_sentence = token
            bert_token = tokenizer.encode(token_text, add_special_tokens=True)[1:-1]

            if bert_token:
                bert_tokens_ids.extend(bert_token)
                start_bert_idx.append(
                    bert_cursor)  # index of the first wordpiece for this token in bert_tokens_ids
                bert_cursor += len(bert_token)
                end_bert_idx.append(
                    bert_cursor - 1)  # index of the last wordpiece for this token in bert_tokens_ids

                sentence_alignment.extend([sent_id] * len(bert_token))  # sentence ID of each wordpiece
                token_alignment.extend([token_id] * len(bert_token))  # token ID of each wordpiece
            else:
                raise ValueError(f"Empty token? What's going on? Document is {doc_id}.")

        # segment intervals are [inclusive, exclusive]!
        segments = Corpus._split_doc_into_segments(token_alignment,
                                                   sentence_alignment,
                                                   max_segment_length=segment_window - 2)
        token_ids = [x[1] for x in tokens]
        bert_segments, original_segments, start_end_segment = [], [], []
        delta = 0

        for start, end in zip(segments, segments[1:]):
            original_start = token_ids.index(token_alignment[start])  # index (in original list of tokens) of the token at the start of the segment
            original_end = token_ids.index(token_alignment[end - 1])  # index (in original list of tokens) of the token at the end of the segment

            # get the indices of the first ("start") and last ("end") wordpieces in the BERT token IDs for each token
            bert_start = np.array(start_bert_idx[original_start:original_end + 1]) - delta
            bert_end = np.array(end_bert_idx[original_start:original_end + 1]) - delta

            original_segments.append(tokens[original_start:original_end + 1])
            bert_ids = tokenizer.encode(' '.join([x[2] for x in tokens[original_start:original_end + 1]]),
                                        add_special_tokens=True)[1:-1]

            if len(bert_ids) != (end - start):
                raise ValueError(doc_id, start, end, len(bert_ids), (end - start))
            elif bert_start.size == 0 or bert_end.size == 0:
                raise ValueError(f"Empty bert_start or bert_end: {doc_id}, {bert_start}, {bert_end}")

            bert_segments.append(bert_ids)
            start_end = np.concatenate((np.expand_dims(bert_start, 1),
                                        np.expand_dims(bert_end, 1)), axis=1)
            start_end_segment.append(start_end)
            delta = end
        segment_doc = [doc_id] * (len(segments) - 1)
        return start_end_segment, segment_doc, bert_segments, original_segments


    def bert_tokenize(self, tokenizer, segment_window: int):
        """
        BERT tokenize documents and split them into segments.
        """
        self.list_of_docs = {}
        self.origin_tokens = {}
        self.bert_tokens = {}
        self.start_end_bert = {}
        self.mentions_to_segment_ids = {}

        for topic, doc_ids in tqdm.tqdm(self.docs_by_topic.items(), desc="Tokenizing", unit="topic", mininterval=10):
            segments_doc_ids = []
            segments_bert_tokens = []
            segments_origin_tokens = []
            segments_bert_start_end = []
            topic_mentions_to_segment_ids = collections.defaultdict(dict)

            for doc_id in doc_ids:
                doc_segments_bert_start_end, doc_segments_doc_ids, doc_segments_bert_tokens, doc_segments_original_tokens = Corpus._bert_tokenize_document(doc_id, self.documents[doc_id], tokenizer, segment_window)

                # map mention to the right segment idx to simplify batch creation later on
                for span_origin_start, span_origin_end in self.mentions_with_cluster_id[topic][doc_id].keys():
                    for segment_idx, origin_tokens in enumerate(doc_segments_original_tokens):
                        min_token_id_in_segment, max_token_id_in_segment = origin_tokens[0][1], origin_tokens[-1][1]
                        mention_starts_in_segment = span_origin_start >= min_token_id_in_segment and span_origin_start <= max_token_id_in_segment
                        mention_ends_in_segment = span_origin_end <= max_token_id_in_segment
                        if mention_starts_in_segment != mention_ends_in_segment:
                            raise ValueError(f"Oh noez: mention {doc_id}, start {span_origin_start}, end {span_origin_end} was split across two segments.")
                        elif mention_starts_in_segment and mention_ends_in_segment:
                            # we need to offset the resulting index by the number of segments of all documents segmented previously in this topic...
                            topic_mentions_to_segment_ids[doc_id][(span_origin_start, span_origin_end)] = len(segments_doc_ids) + segment_idx
                            break

                segments_doc_ids += doc_segments_doc_ids
                segments_bert_start_end += doc_segments_bert_start_end
                segments_bert_tokens += doc_segments_bert_tokens
                segments_origin_tokens += doc_segments_original_tokens

            self.list_of_docs[topic] = segments_doc_ids
            self.origin_tokens[topic] = segments_origin_tokens
            self.bert_tokens[topic] = segments_bert_tokens
            self.start_end_bert[topic] = segments_bert_start_end
            self.mentions_to_segment_ids[topic] = topic_mentions_to_segment_ids

    @staticmethod
    def create_corpus(config, tokenizer, split_name, is_training=True) -> "Corpus":
        docs_path = os.path.join(config.data_folder, split_name + '.json')
        mentions_path = os.path.join(config.data_folder,
                                     split_name + '_{}.json'.format(config.mention_type))
        with open(docs_path, 'r') as f:
            documents = json.load(f)

        mentions = []
        if is_training or config.use_gold_mentions:
            with open(mentions_path, 'r') as f:
                mentions = json.load(f)

        predicted_topics = None
        if not is_training and config.use_predicted_topics:
            with open(config.predicted_topics_path, 'rb') as f:
                predicted_topics = pickle.load(f)

        logging.info('Split - {}'.format(split_name))

        corpus = Corpus(documents, mentions, subtopic=config.subtopic, predicted_topics=predicted_topics)
        corpus.bert_tokenize(tokenizer, config.segment_window)
        return corpus