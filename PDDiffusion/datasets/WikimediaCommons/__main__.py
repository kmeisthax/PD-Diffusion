from PDDiffusion.datasets.WikimediaCommons import Connection, PD_ART_US_EXPIRATION_CATEGORY, LOCAL_STORAGE
from urllib.request import urlopen
import os.path, json

conn = Connection()

limit = 200
count = 0

for item in conn.walk_category(PD_EXPIRATION_CATEGORY, member_types=["file"]):
    if count >= limit:
        break
    
    count += 1
    
    print(item["title"])

    localfile = os.path.join(LOCAL_STORAGE, item["title"].removeprefix("File:"))

    if not os.path.exists(LOCAL_STORAGE):
        os.makedirs(LOCAL_STORAGE)
    
    image_info = conn.image_info(titles=[item["title"]], iiprop=["url"])
    for image in image_info["query"]["pages"][str(item["pageid"])]["imageinfo"]:
        with conn.urlopen(image["url"]) as source:
            with open(localfile, "wb") as sink:
                sink.write(source.read())
    
    page_terms = conn.page_terms(titles=[item["title"]])
    if "terms" in page_terms["query"]["pages"][str(item["pageid"])]:
        with open(localfile + '.json', 'w') as sink:
            sink.write(json.dumps(page_terms["query"]["pages"][str(item["pageid"])]["terms"]))
    
    #TODO: Retain category information when crawling.