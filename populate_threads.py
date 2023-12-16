#!/usr/bin/env python

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import time
import os
import re
import sqlite3

def get_pickup_urls(session):
    response = requests.get("https://news.yahoo.co.jp/topics/top-picks")
    time.sleep(1)
    soup = BeautifulSoup(response.text, 'html.parser')
    urls = [link.get('href') for link in soup.find_all(
        "a", class_="newsFeed_item_link")]
    return urls

def get_article_url(session, pickup_url):
    response = session.get(pickup_url)
    time.sleep(1)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.find_all('a', class_="sc-eIRKgS")[0].get("href")

def get_article_contents(session, article_url):
    response = session.get(article_url)
    time.sleep(1)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find_all(
        'h1', class_='sc-fnebDD bAGyCr')[0].get_text(strip=True)
    body = [p.get_text(separator='\n', strip=True) for p in soup.find_all(
        'p', class_='sc-gLMgcV EJLaQ yjSlinkDirectlink highLightSearchTarget')]
    body = [re.sub(r'【写真】.+$', '', line) for line in body]
    return (title, '\n'.join(body))

def create_tables(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS thread
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT)''')

    conn.execute('''CREATE TABLE IF NOT EXISTS post
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER,
                    body TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(thread_id) REFERENCES thread(id))''')

def create_thread(conn, title, body):
    conn.execute("INSERT INTO thread (title) VALUES (?)", (title,))
    conn.execute("INSERT INTO post (thread_id, body) VALUES (last_insert_rowid(), ?)", (body,))
    conn.commit()

conn = sqlite3.connect('database.db')
create_tables(conn)

session = requests.Session()

article_urls = set([get_article_url(session, pickup_url) for pickup_url in tqdm(get_pickup_urls(session))])

for article_url in tqdm(article_urls):
    title, body = get_article_contents(session, article_url)
    create_thread(conn, title, body)

conn.close()