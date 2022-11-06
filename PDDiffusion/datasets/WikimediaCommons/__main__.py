from PDDiffusion.datasets.WikimediaCommons import Connection, PD_ART_US_EXPIRATION_CATEGORY, LOCAL_STORAGE
from urllib.request import urlopen
import os.path, json

conn = Connection()

limit = 200
count = 0

for item in conn.walk_category(PD_ART_US_EXPIRATION_CATEGORY, member_types=["file"]):
    if count >= limit:
        break

    sanitized_image_name = item["title"].removeprefix("File:").replace("\"", "").replace("'", "").replace("?", "").replace("!", "").replace("*", "").strip()
    localfile = os.path.join(LOCAL_STORAGE, sanitized_image_name)

    if os.path.exists(localfile): #Don't rescan old images.
        continue
    
    count += 1
    
    print(item["title"])

    if not os.path.exists(LOCAL_STORAGE):
        os.makedirs(LOCAL_STORAGE)
    
    image_info = conn.image_info(titles=[item["title"]], iiprop=["url"])
    for image in image_info["query"]["pages"][str(item["pageid"])]["imageinfo"]:
        with conn.urlopen(image["url"]) as source:
            with open(localfile, "wb") as sink:
                sink.write(source.read())

    metadata = {}
    metadata["title"] = item["title"].removeprefix("File:").removesuffix(".jpg").removesuffix(".jpeg").removesuffix(".png").removesuffix(".tif").removesuffix(".tiff")

    page_terms = conn.page_terms(titles=[item["title"]])
    if "terms" in page_terms["query"]["pages"][str(item["pageid"])]:
        metadata["terms"] = page_terms["query"]["pages"][str(item["pageid"])]["terms"]
    
    with open(localfile + '.json', 'w') as sink:
        sink.write(json.dumps(metadata))
    
    #TODO: Retain category information when crawling.