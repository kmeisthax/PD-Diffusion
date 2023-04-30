from PDDiffusion.datasets.WikimediaCommons.model import WikimediaCommonsImage
from PDDiffusion.datasets.WikimediaCommons import Connection, DEFAULT_UA, BASE_API_ENDPOINT, PD_ART_CATEGORY_OLD100, LOCAL_STORAGE, scrape_and_save_metadata, chunked_iterable
from PDDiffusion.datasets.model import Dataset
import os.path, sys, itertools
from dataclasses import field
from argparse_dataclass import dataclass
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

@dataclass
class WikimediaScrapeOptions:
    email:str = field(metadata={"help": "E-mail address to attach to all requests, e.g. --email test@example.com"})
    limit:int = field(metadata={"help": "How many new images to download"}, default=None)
    endpoint:str = field(metadata={"help": "Which Wiki mirror to scrape from, default is Wikimedia Commons"}, default=BASE_API_ENDPOINT)
    ua:str = field(metadata={"args": ["--user-agent"], "help": "Set the user-agent string for all requests"}, default=DEFAULT_UA)
    update:bool = field(metadata={"args": ["--update"], "help": "Check already-downloaded files for updates. (Incompatible with subcategory storage.)"}, default=False)
    rescrape:bool = field(metadata={"args": ["--rescrape"], "help": "Redownload metadata for already-downloaded files and categories"}, default=False)
    files:bool = field(metadata={"args": ["--files"], "help": "Download files that are children of the given category"}, default=False)
    subcategories:bool = field(metadata={"args": ["--subcategories"], "help": "Store all categories traversed when downloading images. This is also necessary for category exclusion or weighting during dataset export."}, default=False)
    category:str = field(metadata={"args": ["--category"], "help": "Change the starting category to scrape from"}, default=PD_ART_CATEGORY_OLD100)

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

    if options.update:
        if options.subcategories:
            raise Exception("Update and subcategory scrape are mutually exclusive options")
        
        source = WikimediaCommonsImage.select_all_image_articles(session)
    else:
        member_types = []
        if options.files:
            member_types.append("file")
        
        if options.subcategories:
            member_types.append("subcat")

        source = conn.walk_category(options.category, member_types=member_types)
    
    if options.subcategories:
        count += scrape_and_save_metadata(conn, session, pages=[options.category], force_rescrape=options.rescrape)
    
    for pages in chunked_iterable(source, 10):
        if options.limit is not None and count >= options.limit:
            break

        this_count = scrape_and_save_metadata(conn, session, pages=pages, force_rescrape=options.rescrape)
        if this_count > 0:
            session.commit()
        
        count += this_count
    
    session.commit()