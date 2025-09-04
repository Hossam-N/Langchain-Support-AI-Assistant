import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time
import argparse
import json
from pathlib import Path
import pdb


class BaseScrapper:
    def __init__(self, base_url: str, max_depth: int = 2, delay: float = 0.5):
        self.base_url = base_url
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls = set()
        self.docs = []

    def scrape_page(self, url: str) -> str:
        try:
            respone = requests.get(url, timeout=10)
            respone.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return ""
        
        soup = BeautifulSoup(respone.text, 'html.parser')
        content = soup.find("article")
        return content.get_text(separator="\n", strip=True) if content else ""

    def crawl(self, url: str, depth: int = 0):
        if depth > self.max_depth or url in self.visited_urls:
            return
        self.visited_urls.add(url)
        
        print(f"Crawling: {url} at depth {depth}")
        text = self.scrape_page(url)
        if text:
            self.docs.append((url, text))
            print(f"[INFO] scraped from {url}")

        try:
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            for link in soup.find_all('a', href=True):
                next_url = urljoin(self.base_url, link['href'])
                if next_url.startswith(self.base_url) and next_url not in self.visited_urls:
                    time.sleep(self.delay)
                    self.crawl(next_url, depth + 1)    
        except requests.RequestException as e:
            print(f"[WARN] Failed to retrieve links from {url}: {e}")


    def save_to_json(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(self.docs)} documents → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Scrape Langchain Documentaion")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/langchain_docs.json",
        help="Path to save scraped documents as JSON"
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Maximum crawl depth"
    )

    args = parser.parse_args()
    BASE_URL = "https://python.langchain.com/docs/"
    scrapper = BaseScrapper(BASE_URL, max_depth= args.depth)
    # pdb.set_trace()
    scrapper.crawl(BASE_URL)
    scrapper.save_to_json(args.output)





        
        