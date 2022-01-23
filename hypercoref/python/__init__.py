# common crawl data constants
URL = "url"
TIMESTAMP = "timestamp"
DIGEST = "digest"
FILENAME = "filename"
OFFSET = "offset"
LENGTH = "length"

# files produced by pipeline stages
PAGE_INDEX = "page_index"
PAGE_INFOS = "page_infos"
SENTENCES = "sentences" # cleaned page content, sentence tokenized
HYPERLINKS = "hyperlinks"
TOKENS = "tokens"       # cleaned page content, word tokenized

# news article data constants
PUBLISH_DATE = "publish-date"
CONTENT_CHARS = "content-chars"
TITLE = "title"
TEXT = "text"   # cleaned page content, not tokenized
AUTHORS = "authors"
TO_URL = "to-url"
TOP_IMAGE_URL = "top-image-url"
TO_URL_NORMALIZED = "to-url-normalized"     # without scheme, query, fragment
URL_NORMALIZED = "url-normalized"           # without scheme, query, fragment
ANCHOR_TEXT = "anchor-text"

# common dataset constants
COLLECTION = "collection"
TOPIC_ID = "topic-id"
SUBTOPIC = "subtopic"
DOCUMENT_ID = "doc-id"
DOCUMENT_NUMBER = "doc-number"
EVENT = "event"
EVENT_ID = "event-id"
ENTITY = "entity"
ENTITY_ID = "entity-id"
MENTION_ID = "mention-id"
TOKEN = "token"
TOKEN_IDX = "token-idx"
SENTENCE = "sentence"
SENTENCE_IDX = "sentence-idx"
SENTENCE_TYPE = "sentence-type"
MENTION_TYPE = "mention-type"
DESCRIPTION = "description"

# specifics for span-based data
CHARS_START = "chars-start"
CHARS_END = "chars-end"
TOKEN_IDX_FROM = "token-idx-from"
TOKEN_IDX_TO = "token-idx-to"