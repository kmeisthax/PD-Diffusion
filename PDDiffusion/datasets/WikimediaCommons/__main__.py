from PDDiffusion.datasets.WikimediaCommons import Connection, PD_ART_US_EXPIRATION_CATEGORY, LOCAL_STORAGE
from urllib.request import urlopen
import os.path, json

conn = Connection()

limit = 200
count = 0

if not os.path.exists(LOCAL_STORAGE):
    os.makedirs(LOCAL_STORAGE)

for item in conn.walk_category(PD_ART_US_EXPIRATION_CATEGORY, member_types=["file"]):
    if count >= limit:
        break

    sanitized_image_name = item["title"].removeprefix("File:").replace("\"", "").replace("'", "").replace("?", "").replace("!", "").replace("*", "").strip()
    localfile = os.path.join(LOCAL_STORAGE, sanitized_image_name)
    metadatafile = localfile + '.json'

    if os.path.exists(localfile + ".banned") or os.path.exists(localfile + ".bannedmetadata"):
        #Skip downloaded files that were banned from the training set due to decoding errors
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
    
    if not file_already_exists:
        image_info = conn.image_info(titles=[item["title"]], iiprop=["url"])
        for image in image_info["query"]["pages"][str(item["pageid"])]["imageinfo"]:
            with conn.urlopen(image["url"]) as source:
                with open(localfile, "wb") as sink:
                    sink.write(source.read())
    
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
        
        with open(metadatafile, 'w') as sink:
            sink.write(json.dumps(metadata))
    
    count += 1