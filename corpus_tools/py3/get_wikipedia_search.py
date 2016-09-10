import util
import os
import wikipedia as wiki
from concurrent.futures import ThreadPoolExecutor

dir_keyword = '../data/GIA_keywords'
dir_output = '../data/wikipedia_content_based_on_GIA_keyword_one_file_per_keyword_search/'

def save_wiki_page(keyword):
    content = None
    title = None
    try:
        print('downloading', keyword)
        page = wiki.page(keyword)
        content = page.content
        url = page.url
        title = page.title
    except wiki.exceptions.DisambiguationError as e:
        print('DisambiguationError', keyword)
    except:
        print('Error', keyword)

    if not content or not title:
        return
    #file_meta.write("%s\t%s\t%s\n" % (keyword, title, url))

    path_output = dir_output + '_'.join(title.replace('/', '__').split()) + '.txt'
    if not os.path.exists(path_output):
        with open(path_output, 'w') as out_wiki:
            for line in content.split('\n'):
                line = ' '.join(map(util.norm_word, line.split()))
                if line:
                    out_wiki.write(line + '\n')

def get_wikipedia_content_based_on_ck_12_keyword_one_file_per_keyword():
    '''
    Get wikipedia page content based on the keywords crawled from the ck-12 website.
    '''

    os.makedirs(dir_output, exist_ok=True)

    path_meta = dir_keyword + '_wiki_meta.tsv'
    #file_meta = open(path_meta, 'w')
    lst_keyword = []
    for f in os.scandir(dir_keyword):
        lst_keyword.extend(open(f.path).readlines())

    wiki.set_lang('ru')

    n_total = len(lst_keyword)
    for index, line in enumerate(lst_keyword):
        keyword = line.strip('\n').lower()
        print(index, n_total, index * 1.0 / n_total, keyword)
        search = wiki.search(keyword)
        print('downloading keywords from', keyword)
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(save_wiki_page, search)

if __name__ == '__main__':
    get_wikipedia_content_based_on_ck_12_keyword_one_file_per_keyword()
