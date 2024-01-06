from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import time
import os

def get_dat_urls_in_url(dir_url):
    response = requests.get(dir_url)
    if not response:
        raise ConnectionError(response)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    dat_names = [tag["href"] for tag in soup.find_all('a') if tag["href"].endswith(".dat")]
    return [(dat_name, dir_url + dat_name) for dat_name in dat_names]

def save_dat_files(dat_list, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    session = requests.Session()
    for (dat_name, dat_url) in dat_list:
        time.sleep(1)
        response = session.get(dat_url)
        response.encoding = 'Shift-JIS'
        if not response:
            raise ConnectionError(response)
        with open(os.path.join(dst_dir, dat_name), "w") as f:
            f.write(response.text)

if True:
    RESULT_DIR = "scraped"
    save_dat_files(tqdm(get_dat_urls_in_url("https://asahi.5ch.net/newsplus/oyster/1495/")), dst_dir=RESULT_DIR)
    save_dat_files(tqdm(get_dat_urls_in_url("https://asahi.5ch.net/newsplus/oyster/1496/")), dst_dir=RESULT_DIR)
    save_dat_files(tqdm(get_dat_urls_in_url("https://asahi.5ch.net/newsplus/oyster/1497/")), dst_dir=RESULT_DIR)
    save_dat_files(tqdm(get_dat_urls_in_url("https://asahi.5ch.net/newsplus/oyster/1498/")), dst_dir=RESULT_DIR)
    save_dat_files(tqdm(get_dat_urls_in_url("https://asahi.5ch.net/newsplus/oyster/1498/")), dst_dir=RESULT_DIR)
else:
    RESULT_DIR = "scraped_val"
    save_dat_files(tqdm(get_dat_urls_in_url("https://asahi.5ch.net/newsplus/oyster/1535/")), dst_dir=RESULT_DIR)