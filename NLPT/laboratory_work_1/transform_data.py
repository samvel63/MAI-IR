import ssl
import json
import logging
import os.path
import threading
from os import listdir
from pathlib import Path
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Avoid an error with the ssl certificate
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
russian_stopwords = stopwords.words('russian')

name = os.path.splitext(os.path.basename(__file__))[0]
home_path = '/Users/samvel/projects/MAI-IR'
wikipedia_data_path = os.path.join(home_path, 'IR', 'laboratory_work_1', 'wikipedia')
transformed_data_path = os.path.join(home_path, 'NLPT', 'laboratory_work_1', 'transformed_data')

bad_chars = {
    ')', '(', '[', ']', '{', '}', '\n', '\t', '\\', '/', '"', '\'', '.',
    ',', '*', '#', '@', '$', '%', '_', '-', '!', '?', '<', '>', '=', '+',
    '-', ':', '?', '|', '~', ';', '±', '&', '^', '№', '—', '»', '«', '€',
    '°с', '’', '“', '„', '…', '­'
}


def transform_data(file_path, logger):
    logger.info(f'threadId={threading.current_thread().ident}, transform_file={file_path}')
    with open(file_path) as file:
        lines = file.readlines()

    data = ''
    for line in lines:
        json_data = json.loads(line)
        transformed_data = json_data['text'].lower()

        for bad_char in bad_chars:
            transformed_data = transformed_data.replace(bad_char, '')

        tokens = word_tokenize(transformed_data, language="russian")
        json_data['text'] = " ".join(token for token in tokens if token not in russian_stopwords)
        data += f'{json.dumps(json_data, ensure_ascii=False)}\n'

    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    article_path = os.path.join(transformed_data_path, folder_name, file_name)
    with open(article_path, 'w', encoding='utf8') as new_file:
        new_file.write(data)


if __name__ == '__main__':
    fmt = '%Y-%m-%d'
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%y/%m/%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    articles_paths = []
    for folder in sorted(listdir(wikipedia_data_path)):
        if not folder.startswith('.'):
            Path(os.path.join(transformed_data_path, folder)).mkdir(parents=True, exist_ok=True)
            articles_paths.append(os.path.join(wikipedia_data_path, folder))

    start = datetime.now()

    for articles_path in articles_paths:
        articles_path_folders = sorted(listdir(articles_path))
        articles = [os.path.join(articles_path, folder) for folder in articles_path_folders]

        threads = []
        for number, article in enumerate(articles):
            thread = threading.Thread(target=transform_data, name=f'thread_{number}', args=(article, logger))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    end = datetime.now()
    logger.info(end - start)
