import random, urllib.parse, sys
from PDDiffusion.datasets.WikimediaCommons import local_wikimedia_base, scrape_and_save_metadata, DEFAULT_UA, BASE_API_ENDPOINT, Connection
from dataclasses import field
from argparse_dataclass import dataclass

@dataclass
class LabelingOptions:
    email:str = field(metadata={"help": "E-mail address to attach to all requests, e.g. --email test@example.com"})
    endpoint:str = field(metadata={"help": "Which Wiki mirror to scrape from, default is Wikimedia Commons"}, default=BASE_API_ENDPOINT)
    ua:str = field(metadata={"args": ["--user-agent"], "help": "Set the user-agent string for all requests"}, default=DEFAULT_UA)

options = LabelingOptions.parse_args(sys.argv[1:])
conn = Connection(base_api_endpoint=options.endpoint, ua=options.ua, email=options.email)

unlabeled_images = []

for item in local_wikimedia_base(load_images=False):
    if "__label" not in item or item["__label"] is None or item["__label"].strip() == "":
        unlabeled_images.append(item)

print(f"{len(unlabeled_images)} images need captions.")
print("Follow the link provided and add a caption to the file.")
print("Once done, press enter to rescrape the item and continue. Or press Q and enter to quit.")

while len(unlabeled_images) > 0:
    item = random.choice(unlabeled_images)
    unlabeled_images.remove(item)
    key = input(f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(item['__pagetitle'].replace(' ', '_'))}")

    if not scrape_and_save_metadata(conn, {"title": item["__pagetitle"], "pageid": item["__pageid"]}, rescrape=True):
        print("Warning: item not scraped")

    if key == "q":
        break