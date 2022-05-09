from pdfminer.pdffont import PDFUnicodeNotDefined, PDFFont


class Font:
    def __init__(self, name: str, pdffont: PDFFont):
        self.name = name
        self.pdffont: PDFFont = pdffont
        self.unicode_to_cid = self._create_revers_lookup()

    def encode(self, text: str) -> bytes:
        _bytes = b''
        length = 2 if self.pdffont.is_multibyte() else 1
        for c in text:
            _bytes += self.unicode_to_cid[c].to_bytes(length, 'big')
        return _bytes

    def decode(self, _bytes: bytes) -> str:
        _unicode = ''
        for cid in self.pdffont.decode(_bytes):
            try:
                _unicode += self.pdffont.to_unichr(cid)
            except PDFUnicodeNotDefined:
                pass
        return _unicode

    def available_characters(self):
        return list(self.unicode_to_cid.keys())

    def _create_revers_lookup(self):
        reverse_lookup = {}

        if cid_to_unicode := getattr(self.pdffont, 'cid2unicode', None):
            for k, v in cid_to_unicode.items():
                reverse_lookup[v] = k

        if unicode_map := getattr(self.pdffont, 'unicode_map'):
            for k, v in unicode_map.cid2unichr.items():
                reverse_lookup[v] = k

        return reverse_lookup

