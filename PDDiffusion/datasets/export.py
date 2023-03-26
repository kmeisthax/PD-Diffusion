"""SQL to Dataset export"""

from PDDiffusion.datasets.model import DatasetImage, DatasetLabel, File
from PDDiffusion.datasets.validity import image_is_valid

from argparse_dataclass import dataclass
from dataclasses import field
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from PIL import Image
import sys, os.path, json

@dataclass
class ExportOptions:
    target_dataset_name: str = field(metadata={"args": ["target_dataset_name"], "help": "Name of the dataset to save"})
    verbose: bool = field(default=False, metadata={"args": ["--verbose"], "help": "Log SQL statements as they execute"})
    rows_per_shard: int = field(default=1000, metadata={"args": ["--rows_per_shard"], "help": "How many images per CSV file"})
    maximum_image_size: int = field(default=512, metadata={"args": ["--maximum_image_size"], "help": "How large the images in the dataset should be"})

options = ExportOptions.parse_args(sys.argv[1:])
load_dotenv()
engine = create_engine(os.getenv("DATABASE_CONNECTION"), echo=options.verbose, future=True)

if not os.path.exists("output"):
    os.makedirs("output")

if not os.path.exists(os.path.join("output", options.target_dataset_name)):
    os.makedirs(os.path.join("output", options.target_dataset_name))

with Session(engine) as session:
    keys = set()
    for (key,) in session.execute(select(DatasetLabel.data_key).group_by(DatasetLabel.data_key)):
        keys.add(key)

    shard_id = 0
    items_in_shard = 0

    shard = None
    
    for image in session.execute(select(DatasetImage).where(DatasetImage.is_banned == False)).scalars().all():
        if items_in_shard > options.rows_per_shard:
            shard_id += 1
            items_in_shard = 0
        
        if items_in_shard == 0:
            if shard is not None:
                shard.close()
            
            shard = open(os.path.join("output", options.target_dataset_name, f"test_{shard_id}.json"), 'w', encoding='utf-8')
            if not os.path.exists(os.path.join("output", options.target_dataset_name, f"test_{shard_id}")):
                os.makedirs(os.path.join("output", options.target_dataset_name, f"test_{shard_id}"))
        
        if image.file is None or image.is_banned:
            continue
        
        if image.file.storage_provider != File.LOCAL_FILE:
            print(f"Non-local file provider {image.file.storage_provider}")
            continue

        extracted = {}

        extracted["dataset"] = image.dataset_id
        extracted["id"] = image.id
        
        for key in keys:
            if key not in extracted:
                extracted[key] = ""
        
        for label in session.execute(select(DatasetLabel).where(DatasetLabel.image_id == image.id)).scalars().all():
            extracted[label.data_key] = label.value
        
        if not image_is_valid(image.file.url):
            print(f"Image {image.id} is corrupt, skipping")
            continue

        localfile = image.id.replace(":", "").replace("\"", "").replace("'", "").replace("?", "").replace("!", "").replace("*", "").strip()
        target_filename = os.path.join("output", options.target_dataset_name, f"test_{shard_id}", localfile)
        target_filename_relative_to_dataset = os.path.join(f"test_{shard_id}", localfile)
        try:
            pil_image = Image.open(image.file.url)
            if pil_image.width > options.maximum_image_size or pil_image.height > options.maximum_image_size:
                pil_image.thumbnail((options.maximum_image_size, options.maximum_image_size))
            
            pil_image.save(target_filename)
            pil_image.close()
        except Exception as e:
            print(f"Image {image.id} failed to resize, got error {e}")
        
        extracted["image"] = target_filename_relative_to_dataset

        json.dump(extracted, shard)
        items_in_shard += 1
    
    if shard is not None:
        shard.close()
    
    print(f"Exported {shard_id * options.rows_per_shard + items_in_shard} items")