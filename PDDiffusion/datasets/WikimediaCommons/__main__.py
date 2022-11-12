from PDDiffusion.datasets.WikimediaCommons import Connection, DEFAULT_UA, BASE_API_ENDPOINT, PD_ART_CATEGORY_OLD100, LOCAL_STORAGE, image_is_valid
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
    metadatafile = localfile + '.json'

    if os.path.exists(localfile + ".banned") or os.path.exists(localfile + ".bannedmetadata"):
        #Skip downloaded files that were banned from the training set.
        #Images can be banned either because they were too large to decode,
        #or because the file could not be decoded in PIL.
        continue

    file_already_exists = os.path.exists(localfile)
    metadata_already_exists = os.path.exists(metadatafile)
    if metadata_already_exists:
        #Check if we need to re-scrape categories or wikitext
        with open(metadatafile, 'r') as source:
            source_data = json.load(source)

            if "categories" not in source_data or "parsetree" not in source_data:
                metadata_already_exists = False
    
    if file_already_exists and metadata_already_exists:
        continue
    
    print(item["title"])
    
    image_info = conn.image_info(titles=[item["title"]], iiprop=["url", "size"])["query"]["pages"][str(item["pageid"])]["imageinfo"]
    image_is_banned = False
    for image in image_info:
        if image["size"] > Image.MAX_IMAGE_PIXELS:
            #Don't even download the file, just mark the metadata as banned
            metadatafile = localfile + '.bannedmetadata'
            image_is_banned = True
            break

        if not file_already_exists:
            with conn.urlopen(image["url"]) as source:
                with open(localfile, "wb") as sink:
                    sink.write(source.read())
            
            if not image_is_valid(localfile):
                os.rename(localfile, localfile + ".banned")
                image_is_banned = True
    
    if not metadata_already_exists:
        metadata = {}
        metadata["title"] = item["title"].removeprefix("File:").removesuffix(".jpg").removesuffix(".jpeg").removesuffix(".png").removesuffix(".tif").removesuffix(".tiff")

        page_terms = conn.page_terms(titles=[item["title"]])
        if "terms" in page_terms["query"]["pages"][str(item["pageid"])]:
            metadata["terms"] = page_terms["query"]["pages"][str(item["pageid"])]["terms"]
        
        page_cats = conn.categories(titles=[item["title"]])
        if "categories" in page_cats["query"]["pages"][str(item["pageid"])]:
            metadata["categories"] = page_cats["query"]["pages"][str(item["pageid"])]["categories"]
        
        page_text = conn.parse_tree(item["title"])
        if "parsetree" in page_text["parse"]:
            metadata["parsetree"] = page_text["parse"]["parsetree"]
        
        metadata["imageinfo"] = image_info
        
        with open(metadatafile, 'w') as sink:
            sink.write(json.dumps(metadata))
    
    if not image_is_banned:
        count += 1