from PDDiffusion.WikimediaCommons import Connection, PD_EXPIRATION_CATEGORY

conn = Connection()

for item in conn.walk_category(PD_EXPIRATION_CATEGORY, member_types=["file"]):
    print(item["title"])
    
    image_info = conn.image_info(titles=[item["title"]], iiprop=["url"])
    for image in image_info["query"]["pages"][str(item["pageid"])]["imageinfo"]:
        print (image["url"])
    
    page_terms = conn.page_terms(titles=[item["title"]])
    if "terms" in page_terms["query"]["pages"][str(item["pageid"])]:
        for (key, values) in page_terms["query"]["pages"][str(item["pageid"])]["terms"].items():
            for value in values:
                print("{0}: {1}".format(key, value))