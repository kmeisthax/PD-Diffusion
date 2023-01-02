from PDDiffusion.datasets.WikimediaCommons.model import WikimediaCommonsImage
from PDDiffusion.datasets.WikimediaCommons import Connection, DEFAULT_UA, BASE_API_ENDPOINT, PD_ART_CATEGORY_OLD100, LOCAL_STORAGE, scrape_and_save_metadata
from PDDiffusion.datasets.model import Dataset
import os.path, sys, itertools
from dataclasses import field
from argparse_dataclass import dataclass
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

@dataclass
class WikimediaScrapeOptions:
    email:str = field(metadata={"help": "E-mail address to attach to all requests, e.g. --email test@example.com"})
    limit:int = field(metadata={"help": "How many new images to download"}, default=200)
    endpoint:str = field(metadata={"help": "Which Wiki mirror to scrape from, default is Wikimedia Commons"}, default=BASE_API_ENDPOINT)
    ua:str = field(metadata={"args": ["--user-agent"], "help": "Set the user-agent string for all requests"}, default=DEFAULT_UA)
    update:bool = field(metadata={"args": ["--update"], "help": "Update outdated or damaged images"}, default=False)
    rescrape:bool = field(metadata={"args": ["--rescrape"], "help": "Redownload existing metadata"}, default=False)

options = WikimediaScrapeOptions.parse_args(sys.argv[1:])

load_dotenv()

conn = Connection(base_api_endpoint=options.endpoint, ua=options.ua, email=options.email)
engine = create_engine(os.getenv("DATABASE_CONNECTION"), future=True)

with Session(engine) as session:
    count = 0

    if not os.path.exists(LOCAL_STORAGE):
        os.makedirs(LOCAL_STORAGE)
    
    dataset_id = f"WikimediaCommons:{conn.base_api_endpoint}"
    
    if session.execute(select(Dataset).where(Dataset.id == dataset_id)).one_or_none() is None:
        session.add(Dataset(id=dataset_id))

    if options.rescrape or options.update:
        source = WikimediaCommonsImage.select_all_image_articles(session)
    else:
        source = conn.walk_category(PD_ART_CATEGORY_OLD100, member_types=["file"])
    
    for pages in chunked_iterable(source, 10):
        if count >= options.limit:
            break

        count += scrape_and_save_metadata(conn, session, pages=pages, force_rescrape=options.rescrape)
        session.commit()