"""Utility to transfer Wikimedia Commons data from the flat-file system to SQL."""

from PDDiffusion.datasets.WikimediaCommons.wikiparse import extract_information_from_wikitext
from PDDiffusion.datasets.WikimediaCommons.model import WikimediaCommonsImage
from PDDiffusion.datasets.WikimediaCommons import LOCAL_STORAGE, BASE_API_ENDPOINT
from PDDiffusion.datasets.model import Dataset, DatasetImage, File, DatasetLabel
from argparse_dataclass import dataclass
from dataclasses import field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import sys, os.path, glob, json, pathlib, dateutil.parser

@dataclass
class TransferOptions:
    verbose: bool = field(default=False, metadata={"args": ["--verbose"], "help": "Log SQL statements as they execute"})

options = TransferOptions.parse_args(sys.argv[1:])
load_dotenv()
engine = create_engine(os.getenv("DATABASE_CONNECTION"), echo=options.verbose, future=True)

with Session(engine) as session:
    ds = Dataset(id=f"WikimediaCommons:{BASE_API_ENDPOINT}")
    session.add(ds)

    for file in glob.iglob(os.path.join(LOCAL_STORAGE, "*.json")):
        with open(file, "r") as metadata:
            metadata_obj = json.load(metadata)

            if "item" not in metadata_obj:
                print (f"WARN: Legacy file {metadata_obj['title']} was not rescraped and should be deleted")
                continue

            #TODO: Validate if file actually exists or not.
            file_row = File(storage_provider=File.LOCAL_FILE, url=os.path.join(os.path.dirname(file), pathlib.Path(file).stem))
            session.add(file_row)
            session.flush()

            ds_image = DatasetImage(id=metadata_obj["item"]["title"], dataset_id=ds.id, file_id=file_row.id)
            session.add(ds_image)

            wiki_image = WikimediaCommonsImage(
                id=ds_image.id, dataset_id=ds_image.dataset_id,
                post_id=metadata_obj["item"]["pageid"],
                last_edited=dateutil.parser.isoparse(metadata_obj["timestamp"]),
                wikidata=metadata_obj)
            session.add(wiki_image)

            extracts = {}
            for name in metadata_obj["parsetree"]:
                xmlstr = metadata_obj["parsetree"][name]
                extracts.update(extract_information_from_wikitext(xmlstr))
            
            for key in extracts.keys():
                wiki_data = DatasetLabel(image_id=ds_image.id, dataset_id=ds_image.dataset_id, data_key=key, value=extracts[key])
                session.add(wiki_data)
    
    session.commit()