import os
import shutil

dataset_dir = '../data/wikipedia_content_based_on_GIA_keyword_one_file_per_keyword_search'
filtered_dir = '../data/wikipedia_content_based_on_GIA_keyword_one_file_per_keyword_search_clean'

if __name__ == '__main__':
    input_count, filtered_count = 0, 0
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            input_count += 1
            if 'Олимпи' in f or 'Евровиде' in f:
                continue
            filtered_count += 1
            f_path = os.path.join(root, f)
            rel_path = os.path.relpath(f_path, dataset_dir)
            inner_dir = os.path.split(rel_path)[0]

            output_dir = os.path.join(filtered_dir, inner_dir)
            os.makedirs(output_dir, exist_ok=True)

            shutil.copy(f_path, output_dir)
    print('found {} files, {} files left'.format(input_count, filtered_count))
