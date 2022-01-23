

class DocumentCleaner:

    def load_markup(self, url, markup):
        self._load_markup(url, markup)

    def _load_markup(self, url, markup):
        raise NotImplementedError

    @property
    def cleaned_text(self):
        raise NotImplementedError

    @property
    def top_node(self):
        """
        etree: The top Element that is a candidate for the main body of the article
        :return:
        """
        raise NotImplementedError

    @property
    def title(self):
        raise NotImplementedError

    @property
    def publish_date(self):
        raise NotImplementedError

    @property
    def authors(self):
        raise NotImplementedError

    @property
    def top_image_url(self):
        raise NotImplementedError