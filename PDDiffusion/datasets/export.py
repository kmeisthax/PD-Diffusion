"""SQL to Dataset export"""

from PDDiffusion.datasets.model import DatasetImage, DatasetLabel, File
from PDDiffusion.datasets.validity import image_is_valid

from argparse_dataclass import dataclass
from dataclasses import field
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from PIL import Image
from multiprocessing import Pool, TimeoutError
import sys, os.path, json, threading

def resize_image_process(open_location, save_location, size):
    result_object = {}

    try:
        if not image_is_valid(open_location):
            result_object["failure"] = f"Image {image.id} is corrupt, skipping"
            return result_object

        image = Image.open(open_location)
        if image.width > size or image.height > size:
            image.thumbnail((size, size))
        
        image.save(save_location)
        image.close()
        
        result_object["success"] = True
        return result_object
    except Exception as e:
        result_object["failure"] = e
        return result_object

@dataclass
class ExportOptions:
    target_dataset_name: str = field(metadata={"args": ["target_dataset_name"], "help": "Name of the dataset to save"})
    verbose: bool = field(default=False, metadata={"args": ["--verbose"], "help": "Log SQL statements as they execute"})
    rows_per_shard: int = field(default=1000, metadata={"args": ["--rows_per_shard"], "help": "How many images per CSV file"})
    maximum_image_size: int = field(default=512, metadata={"args": ["--maximum_image_size"], "help": "How large the images in the dataset should be"})

class AsyncShardCloseThread(threading.Thread):
    """Thread that closes out a given shard.
    
    Closing involves making sure all resize threads have completed and data has
    been written to disk."""
    def __init__(self, encountered_items, shard_save_location, verbose, id):
        """Create the async close thread.
        
        The list of encountered items should be a list of 2-tuples, each one
        containing the following items:
        
         - The data item to save
         - The process pool's AsyncResult, which will return the result of the
         associated resize operation
        
        shard_save_location should be a valid path to save the shard to.
        verbose is whether or not we want database stuff to be printed.
        id is our shard number."""
        super(AsyncShardCloseThread, self).__init__()

        self.encountered_items = encountered_items
        self.shard = open(os.path.join(shard_save_location, f"train_{id}.json"), 'w', encoding='utf-8')
        self.verbose = verbose
        self.id = id
    
    def run(self):
        engine = create_engine(os.getenv("DATABASE_CONNECTION"), echo=self.verbose, future=True)

        with Session(engine) as session:
            for (item, resize_result) in self.encountered_items:
                try:
                    resize_result = resize_result.get(5*60)
                except TimeoutError as e:
                    resize_result = {"failure": f"Child process did not complete within 5 minutes"}
                except Exception as e:
                    resize_result = {"failure": f"Child process failed with {e}"}
                
                if "success" in resize_result:
                    for label in session.execute(select(DatasetLabel).where(DatasetLabel.image_id == item["id"])).scalars().all():
                        item[label.data_key] = label.value

                    json.dump(item, self.shard)
                    self.shard.write("\n")
                else:
                    print(f"Image {item['id']} could not be resized, got error {resize_result['failure']}")

            self.shard.close()
        
        print(f"Closed out shard {self.id}")

if __name__ == "__main__":
    options = ExportOptions.parse_args(sys.argv[1:])
    load_dotenv()
    engine = create_engine(os.getenv("DATABASE_CONNECTION"), echo=options.verbose, future=True)

    if not os.path.exists("output"):
        os.makedirs("output")

    if not os.path.exists(os.path.join("output", options.target_dataset_name)):
        os.makedirs(os.path.join("output", options.target_dataset_name))

    with Session(engine) as session:
        with Pool() as pool:
            keys = set()
            for (key,) in session.execute(select(DatasetLabel.data_key).group_by(DatasetLabel.data_key)):
                keys.add(key)

            shard_id = 0
            items_in_shard = 0

            encountered_items = []
            open_shards = []

            def close_last_shard():
                global encountered_items
                
                shard_thread = AsyncShardCloseThread(encountered_items, os.path.join("output", options.target_dataset_name), options.verbose, shard_id)
                open_shards.append(shard_thread)
                shard_thread.start()

                encountered_items = []
            
            for image in session.execute(select(DatasetImage).where(DatasetImage.is_banned == False)).scalars().all():
                if items_in_shard > options.rows_per_shard:
                    close_last_shard()

                    shard_id += 1
                    items_in_shard = 0
                
                if not os.path.exists(os.path.join("output", options.target_dataset_name, f"train_{shard_id}")):
                    os.makedirs(os.path.join("output", options.target_dataset_name, f"train_{shard_id}"))
                
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

                localfile = image.id.replace(":", "").replace("\"", "").replace("'", "").replace("?", "").replace("!", "").replace("*", "").strip()
                target_filename = os.path.join("output", options.target_dataset_name, f"train_{shard_id}", localfile)
                target_filename_relative_to_dataset = os.path.join(f"train_{shard_id}", localfile)
                
                extracted["image"] = target_filename_relative_to_dataset

                process_result = pool.apply_async(resize_image_process, [image.file.url, target_filename, options.maximum_image_size])
                encountered_items.append((extracted, process_result))

                items_in_shard += 1
            
            close_last_shard()

            for thread in open_shards:
                thread.join()
            
            print(f"Exported {shard_id * options.rows_per_shard + items_in_shard} items")