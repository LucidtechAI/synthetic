from pdfminer.pdffont import PDFFont


class Font:
    def __init__(self, name: str, pdffont: PDFFont):
        self.name = name
        self.pdffont: PDFFont = pdffont
        self.unicode_to_cid = self._create_revers_lookup()
        self.available_characters = set()

    def encode(self, text: str) -> bytes:
        _bytes = b''
        length = 2 if self.pdffont.is_multibyte() else 1
        for c in text:
            _bytes += self.unicode_to_cid[c].to_bytes(length, 'big')
        return _bytes

    def decode(self, _bytes: bytes) -> str:
        _unicode = ''
        for cid in self.pdffont.decode(_bytes):
            _unicode += self.pdffont.to_unichr(cid)
        return _unicode

    def _create_revers_lookup(self):
        reverse_lookup = {}

        if cid_to_unicode := getattr(self.pdffont, 'cid2unicode', None):
            for k, v in cid_to_unicode.items():
                reverse_lookup[v] = k

        if unicode_map := getattr(self.pdffont, 'unicode_map'):
            for k, v in unicode_map.cid2unichr.items():
                reverse_lookup[v] = k

        # Whitespace will sometimes be encoded as soft-hyphen (173), which does not look correct in the output PDF
        whitespace = ' '
        if reverse_lookup.get(whitespace) == 173:
            reverse_lookup[whitespace] = ord(whitespace)

        return reverse_lookup


