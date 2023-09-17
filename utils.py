import os
import torch
from tqdm import tqdm

PATH = r'books-3500/data'
# counter = 0

for folder in tqdm(os.listdir(PATH)[6:7]):
    # if folder not in ['01']:
    for folder_ in os.listdir(os.path.join(PATH, folder)):
        for file in os.listdir(os.path.join(PATH, folder, folder_))[:50]:
            new_file = open('corpus/corpus_2.txt', 'a', encoding='UTF-8')
            if os.path.isfile(os.path.join(PATH, folder, folder_, file)):
                # if it is a file
            # counter+=1
                with open(os.path.join(PATH, folder, folder_, file), 'r', encoding='UTF-8') as f:
                    try:
                        lines = f.readlines()
                        # lines = lines.encode()
                        new_file.writelines(lines)
                    except:
                        print(f'{os.path.join(PATH, folder, folder_, file)} has a problem with codecs!')
            elif os.path.isdir(os.path.join(PATH, folder, folder_, file)):
                # if it is a folder
                for file_ in os.listdir(os.path.join(PATH, folder, folder_, file)):
                    with open(os.path.join(PATH, folder, folder_, file, file_), 'r', encoding='UTF-8') as f:
                        try:
                            lines = f.readlines()
                            # lines = lines.encode()
                            new_file.writelines(lines)
                        except:
                            print(f'{os.path.join(PATH, folder, folder_, file, file_)} has a problem with codecs!')

new_file.close()