from typing import Optional

from goose3 import Goose
from goose3.article import Article

from python.common_components.document_cleaner import DocumentCleaner


class GooseDocumentCleaner(DocumentCleaner):

    def __init__(self):
        self.goose = Goose({"pretty_lists": False, "parse_lists": False, "target_language": "en"})

        self.article = None  # type: Optional[Article]

    def _load_markup(self, url, markup):
        self.article = self.goose.extract(raw_html=markup)

    @property
    def cleaned_text(self):
        if self.article is None:
            return None
        return self.article.cleaned_text

    @property
    def top_node(self):
        """
        etree: The top Element that is a candidate for the main body of the article
        :return:
        """
        if self.article is None:
            return None
        return self.article.top_node

    @property
    def title(self):
        if self.article is None:
            return None
        return self.article.title

    @property
    def publish_date(self):
        if self.article is None:
            return None
        return self.article.publish_datetime_utc

    @property
    def authors(self):
        if self.article is None:
            return None
        return self.article.authors

    @property
    def top_image_url(self):
        return None