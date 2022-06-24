import argparse
import glob
import json
import pathlib
from pdfminer.high_level import extract_text


WORDS_FILE = pathlib.Path(__file__).parent / 'words.json'


def main(args):
    file_path = args.file_path
    frequency = int(args.min_frequency)

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
                if not word.isnumeric():
                    word_dict[word] = word_dict.get(word, 0) + 1

    word_list = []
    for word, word_frequency in word_dict.items():
        if word_frequency >= frequency:
            word_list.append(word)

    with open(WORDS_FILE, 'w') as words_file:
        json.dump(word_list, words_file)
    print(f'Written {len(word_list)} words to {WORDS_FILE}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='path to folder containing the files from which to create wordlist')
    parser.add_argument('min_frequency', help='minimal frequency for which to add word to word.json')
    arguments = parser.parse_args()

    main(arguments)
