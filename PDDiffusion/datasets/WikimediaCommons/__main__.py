from PDDiffusion.datasets.WikimediaCommons import Connection, DEFAULT_UA, BASE_API_ENDPOINT, PD_ART_CATEGORY_OLD100, LOCAL_STORAGE, scrape_and_save_metadata
from urllib.request import urlopen
import os.path, json, sys
from PIL import Image
from dataclasses import field
from argparse_dataclass import dataclass

@dataclass
class WikimediaScrapeOptions:
    email:str = field(metadata={"help": "E-mail address to attach to all requests, e.g. --email test@example.com"})
    limit:int = field(metadata={"help": "How many new images to download"}, default=200)
    endpoint:str = field(metadata={"help": "Which Wiki mirror to scrape from, default is Wikimedia Commons"}, default=BASE_API_ENDPOINT)
    ua:str = field(metadata={"args": ["--user-agent"], "help": "Set the user-agent string for all requests"}, default=DEFAULT_UA)
    rescrape:str = field(metadata={"args": ["--rescrape"], "help": "Redownload existing metadata or files if changed"})

options = WikimediaScrapeOptions.parse_args(sys.argv[1:])

conn = Connection(base_api_endpoint=options.endpoint, ua=options.ua, email=options.email)

count = 0

if not os.path.exists(LOCAL_STORAGE):
    os.makedirs(LOCAL_STORAGE)

for item in conn.walk_category(PD_ART_CATEGORY_OLD100, member_types=["file"]):
    if count >= options.limit:
        break

    sanitized_image_name = item["title"].removeprefix("File:").replace("\"", "").replace("'", "").replace("?", "").replace("!", "").replace("*", "").strip()
    localfile = os.path.join(LOCAL_STORAGE, sanitized_image_name)
    
    if scrape_and_save_metadata(conn, item, localfile):
        count += 1