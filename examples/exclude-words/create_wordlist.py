import argparse
import glob
import json
import pathlib
from pdfminer.high_level import extract_text


def main(args):
    file_path = args.file_path
    frequency = int(args.minimal_frequency)
    minimal_length = int(args.minimal_length)
    words_file = args.words_file

    if not file_path[-1] == '/':
        file_path += '/'

    word_dict = {}
    for file in glob.glob(file_path + '*.pdf'):
        text = extract_text(file)
        sentence_list = text.split('\n')
        sentence_list = [word for word in sentence_list if word != '']

        for sentence in sentence_list:
            for word in sentence.split(' '):
                word = word.lower()
                if len(word) > 1 and word[-1] == ':':
                    word = word[:-1]
                if not word.isnumeric() and len(word) >= minimal_length:
                    word_dict[word] = word_dict.get(word, 0) + 1

    word_list = []
    for word, word_frequency in word_dict.items():
        if word_frequency >= frequency:
            word_list.append(word)

    with open(words_file, 'w') as json_file:
        json.dump(word_list, json_file)
    print(f'Written {len(word_list)} words to {words_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create wordlist of most common words from files in a folder. These will not be changed '
        'when documents are synthesized.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file_path', help='path to folder containing the files from which to create wordlist')
    parser.add_argument('minimal_frequency', help='minimal frequency for which to add word to word.json')
    parser.add_argument('--minimal_length', help='minimal length of words added to the word list', default=3)
    parser.add_argument('--words_file',
                        help='json file containing the wordlist',
                        default=pathlib.Path(__file__).parent / 'words.json')
    arguments = parser.parse_args()

    main(arguments)
