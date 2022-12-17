from PDDiffusion.datasets.WikimediaCommons import Connection, DEFAULT_UA, BASE_API_ENDPOINT, PD_ART_CATEGORY_OLD100, LOCAL_STORAGE, scrape_and_save_metadata
import os.path, sys, itertools, glob
from dataclasses import field
from argparse_dataclass import dataclass

@dataclass
class WikimediaScrapeOptions:
    email:str = field(metadata={"help": "E-mail address to attach to all requests, e.g. --email test@example.com"})
    limit:int = field(metadata={"help": "How many new images to download"}, default=200)
    endpoint:str = field(metadata={"help": "Which Wiki mirror to scrape from, default is Wikimedia Commons"}, default=BASE_API_ENDPOINT)
    ua:str = field(metadata={"args": ["--user-agent"], "help": "Set the user-agent string for all requests"}, default=DEFAULT_UA)
    rescrape:bool = field(metadata={"args": ["--rescrape"], "help": "Redownload existing metadata"}, default=False)

options = WikimediaScrapeOptions.parse_args(sys.argv[1:])

conn = Connection(base_api_endpoint=options.endpoint, ua=options.ua, email=options.email)

count = 0

if not os.path.exists(LOCAL_STORAGE):
    os.makedirs(LOCAL_STORAGE)

if options.rescrape:
    for localfile in itertools.chain(
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpg")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpeg")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.png")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.tif")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.tiff"))
        ):
        if count >= options.limit:
            break

        if scrape_and_save_metadata(conn, localfile=localfile, rescrape=False):
            count += 1
else:
    for item in conn.walk_category(PD_ART_CATEGORY_OLD100, member_types=["file"]):
        if count >= options.limit:
            break
        
        if scrape_and_save_metadata(conn, item, rescrape=False):
            count += 1