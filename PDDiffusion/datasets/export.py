"""SQL to Dataset export"""

from PDDiffusion.datasets.model import DatasetImage, DatasetLabel, File
from PDDiffusion.datasets.validity import image_is_valid

from argparse_dataclass import dataclass
from dataclasses import field
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from PIL import Image
import sys, os.path, json, threading

@dataclass
class ExportOptions:
    target_dataset_name: str = field(metadata={"args": ["target_dataset_name"], "help": "Name of the dataset to save"})
    verbose: bool = field(default=False, metadata={"args": ["--verbose"], "help": "Log SQL statements as they execute"})
    rows_per_shard: int = field(default=1000, metadata={"args": ["--rows_per_shard"], "help": "How many images per CSV file"})
    maximum_image_size: int = field(default=512, metadata={"args": ["--maximum_image_size"], "help": "How large the images in the dataset should be"})

class AsyncResizeThread(threading.Thread):
    """Thread that opens an image, resizes it if it's too big, and saves it
    somewhere else.
    
    Results are reported in the given result_object. No locking is performed on
    the object; you must join this thread before accessing the results.
    
    This expects PIL to not hold the CPython GIL during the resize."""
    def __init__(self, open_location, size, save_location, result_object):
        super(AsyncResizeThread, self).__init__()

        self.open_location = open_location
        self.size = size
        self.save_location = save_location
        self.result_object = result_object
    
    def run(self):
        try:
            image = Image.open(self.open_location)
            if image.width > self.size or image.height > self.size:
                image.thumbnail((self.size, self.size))
            
            image.save(self.save_location)
            image.close()
            self.result_object["success"] = True
        except Exception as e:
            self.result_object["failure"] = e

class AsyncShardCloseThread(threading.Thread):
    """Thread that closes out a given shard.
    
    Closing involves making sure all resize threads have completed and data has
    been written to disk."""
    def __init__(self, encountered_items, shard):
        """Create the async close thread.
        
        The list of encountered items should be a list of 3-tuples, each one
        containing the following items:
        
         - The data item to save
         - The resize thread that saved the image already
         - The result object that the thread saves data to
        
        shard should be an open JSON file which we will write data to."""
        super(AsyncShardCloseThread, self).__init__()

        self.encountered_items = encountered_items
        self.shard = shard
    
    def run(self):
        for (item, resize_thread, resize_result) in self.encountered_items:
            resize_thread.join()

            if "success" in resize_result:
                json.dump(item, self.shard)
                self.shard.write("\n")
            else:
                print(f"Image {item['id']} could not be resized, got error {resize_result['failure']}")

        self.shard.close()

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

    encountered_items = []

    def close_last_shard():
        global encountered_items

        if shard is not None:
            AsyncShardCloseThread(encountered_items, shard).start()
            encountered_items = []
    
    for image in session.execute(select(DatasetImage).where(DatasetImage.is_banned == False)).scalars().all():
        if items_in_shard > options.rows_per_shard:
            shard_id += 1
            items_in_shard = 0
        
        if items_in_shard == 0:
            close_last_shard()
            
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
        
        extracted["image"] = target_filename_relative_to_dataset

        result = {}
        thread = AsyncResizeThread(image.file.url, options.maximum_image_size, target_filename, result)
        encountered_items.append((extracted, thread, result))

        thread.start()

        items_in_shard += 1
    
    close_last_shard()
    
    print(f"Exported {shard_id * options.rows_per_shard + items_in_shard} items")