import collections
import unicodedata
import re


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    if not init_checkpoint:
        return

    m = re.match(r"^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16",
        "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12",
        "chinese_L-12_H-768_A-12",
    ]

    cased_models = [
        "cased_L-12_H-768_A-12",
        "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12",
    ]

    if model_name in lower_models and not do_lower_case:
        raise ValueError("Checkpoint expects lowercased model.")

    if model_name in cased_models and do_lower_case:
        raise ValueError("Checkpoint expects cased model.")


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        for index, token in enumerate(reader):
            token = token.strip()
            vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


class FullTokenizer:
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            split_tokens.extend(self.wordpiece_tokenizer.tokenize(token))
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[i] for i in ids]


class BasicTokenizer:
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []

        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._strip_accents(token)
            split_tokens.extend(self._split_on_punc(token))

        return whitespace_tokenize(" ".join(split_tokens))

    def _strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        return "".join(
            char for char in text
            if unicodedata.category(char) != "Mn"
        )

    def _split_on_punc(self, text):
        chars = list(text)
        output = []
        current = []

        for char in chars:
            if _is_punctuation(char):
                if current:
                    output.append("".join(current))
                    current = []
                output.append(char)
            else:
                current.append(char)

        if current:
            output.append("".join(current))

        return output

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []

        for token in whitespace_tokenize(text):
            chars = list(token)

            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None

                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


def _is_whitespace(char):
    if char in [" ", "\t", "\n", "\r"]:
        return True
    return unicodedata.category(char) == "Zs"


def _is_control(char):
    if char in ["\t", "\n", "\r"]:
        return False
    return unicodedata.category(char) in ("Cc", "Cf")


def _is_punctuation(char):
    cp = ord(char)
    if (
        33 <= cp <= 47
        or 58 <= cp <= 64
        or 91 <= cp <= 96
        or 123 <= cp <= 126
    ):
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese_char(cp):
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )