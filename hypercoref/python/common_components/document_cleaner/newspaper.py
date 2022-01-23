from typing import Optional

from newspaper.article import Article, ArticleDownloadState

from python.common_components.document_cleaner import DocumentCleaner


class NewspaperDocumentCleaner(DocumentCleaner):

    def __init__(self):
        self.article = None  # type: Optional[Article]

    def _load_markup(self, url, markup):
        article = Article(url)
        article.download(input_html=markup)

        # Sometimes newspaper3k does not recognize the markup, then it considers an article as not downloaded and fails
        # in the parse() call. Therefore check the article status before calling parse().
        if not article.download_state == ArticleDownloadState.SUCCESS:
            raise ValueError
        article.parse()
        self.article = article

    @property
    def cleaned_text(self):
        if self.article is None:
            return None
        return self.article.text

    @property
    def top_node(self):
        """
        etree: The top Element that is a candidate for the main body of the article
        :return:
        """
        if self.article is None:
            return None
        return self.article.clean_top_node

    @property
    def title(self):
        if self.article is None:
            return None
        return self.article.title

    @property
    def publish_date(self):
        if self.article is None:
            return None
        return self.article.publish_date

    @property
    def authors(self):
        if self.article is None:
            return None
        return self.article.authors

    @property
    def top_image_url(self):
        if self.article is None:
            return None
        return self.article.top_image