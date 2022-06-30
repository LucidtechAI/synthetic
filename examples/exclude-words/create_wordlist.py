import argparse
import filetype
import glob
import json
from collections import Counter
from filetype.types.archive import Pdf
from pathlib import Path
from pdfminer.high_level import extract_text


def main(args):
    file_path = args.file_path
    minimum_frequency = args.minimum_frequency
    minimum_length = args.minimum_length
    allow_numeric = args.allow_numeric
    remove_characters = args.remove_characters
    words_file = args.words_file

    word_dict = Counter()
    for file in file_path.iterdir():
        kind = filetype.guess(file)
        if not isinstance(kind, Pdf):
            continue
        text = extract_text(file)
        sentence_list = text.split('\n')
        sentence_list = [sentence for sentence in sentence_list if sentence]

        for sentence in sentence_list:
            for word in sentence.split(' '):
                word = word.lower()
                word = word.replace(remove_characters, '')
                if not allow_numeric and word.isnumeric():
                    continue
                if len(word) >= minimum_length:
                    word_dict[word] += 1

    word_list = []
    for word, word_frequency in word_dict.items():
        if word_frequency >= minimum_frequency:
            word_list.append(word)

    words_file.write_text(json.dumps(word_list, indent=2))
    print(f'Written {len(word_list)} words to {words_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create wordlist of most common words from files in a folder. These will not be changed '
        'when documents are synthesized.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'file_path', type=Path,
        help='path to folder containing the files from which to create wordlist',
    )
    parser.add_argument('minimum_frequency', type=int, help='minimum frequency for which to add word to word.json')
    parser.add_argument('--minimum_length', type=int, help='minimum length of words added to the word list', default=3)
    parser.add_argument('--allow_numeric', action='store_true', help='let numeric strings be added to word list')
    parser.add_argument('--remove_characters', help='remove these characters from words in wordlist', default='')
    parser.add_argument(
        '--words_file',
        help='json file containing the wordlist',
        type=Path,
        default=Path(__file__).parent / 'words.json',
    )
    arguments = parser.parse_args()

    main(arguments)
