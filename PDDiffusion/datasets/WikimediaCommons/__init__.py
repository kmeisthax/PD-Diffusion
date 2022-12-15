import requests, urllib.request, glob, os.path, itertools, json, PIL
from PIL import Image
from PDDiffusion.datasets.WikimediaCommons.wikiparse import extract_information_from_wikitext

LOCAL_STORAGE = os.path.join("sets", "wikimedia")
LOCAL_RESIZE_CACHE = os.path.join("sets", "wikimedia-cache")
BASE_API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"

DEFAULT_UA = "PD-Diffusion/0.0"

class IdentityError(Exception):
    pass

class Connection(object):
    def __init__(self, base_api_endpoint=BASE_API_ENDPOINT, ua=DEFAULT_UA, email=None):
        if email is None:
            raise IdentityError("Please provide an e-mail address as per Wikimedia guidelines")
        
        if email.endswith("@example.com"):
            raise IdentityError("Please replace the example e-mail address with your own")
        
        self.base_api_endpoint = base_api_endpoint
        self.ua = "{} ({})".format(ua, email)
    
    def get(self, **query_params):
        query_params["format"] = "json"

        return requests.get(self.base_api_endpoint, params=query_params, headers={'user-agent': self.ua}).json()
    
    def walk_category(self, category_name, member_types=["page"], recursive = True):
        """Walk a category on the given wiki API.
        
        Yields member pages or subcategories in the category as returned from
        the API. This includes a title and pageid at minimum.

        Member types lists all the types you care about.
        
        In Recursive mode, all subcategories will also be walked."""

        cmcontinue = None
        my_cmtype = set(member_types)
        if recursive:
            #We always need to ask for subcats in a recursive walk.
            my_cmtype.add("subcat")

        while True:
            page = self.get(action="query", list="categorymembers", cmtitle=category_name, cmlimit=20, cmcontinue=cmcontinue, cmtype="|".join(my_cmtype))

            for item in page["query"]["categorymembers"]:
                if recursive and item["title"].startswith("Category:"):
                    if "subcat" in member_types:
                        #If the user asked for both subcats AND a recursive walk,
                        #we have to yield both the category and its children
                        yield item
                    
                    yield from self.walk_category(item["title"], member_types=member_types, recursive=recursive)
                else:
                    yield item
            
            if "continue" in page and "cmcontinue" in page["continue"]:
                cmcontinue = page["continue"]["cmcontinue"]
            else:
                return
    
    def info(self, titles=[]):
        return self.get(action="query", prop="info", titles="|".join(titles))

    def image_info(self, titles=[], iiprop=[]):
        return self.get(action="query", prop="imageinfo", titles="|".join(titles), iiprop="|".join(iiprop))
    
    def page_terms(self, titles=[]):
        return self.get(action="query", prop="pageterms", titles="|".join(titles))
    
    def categories(self, titles=[]):
        """Get the categories for a given page."""
        return self.get(action="query", prop="categories", titles="|".join(titles), cllimit="max", clshow="hidden")
    
    def parse_tree(self, title):
        return self.get(action="parse", prop="parsetree", page=title)
    
    def revisions(self, titles=[], rvprop=[], rvlimit=100, rvcontinue=None):
        return self.get(action="query", prop="revisions", titles="|".join(titles), rvprop="|".join(rvprop), rvlimit=rvlimit, rvcontinue=rvcontinue)
    
    def urlopen(self, url):
        if not isinstance(url, urllib.request.Request):
            url = urllib.request.Request(url)
        
        url.add_header("User-Agent", self.ua)

        return urllib.request.urlopen(url)

#ALL Public Domain images.
#Includes things that are only PD in some countries.
#Guaranteed to be at least safe for US use as Wikimedia is US-based.
PD_CATEGORY = "Category:Public domain"

#PD through expiration only.
#Some countries do not recognize public domain dedication and thus
#limiting ourselves to this category is useful.
PD_EXPIRATION_CATEGORY = "Category:Public_domain_due_to_copyright_expiration"

#Uncopyrightable photograph of PD art.
#In the US, you cannot recopyright PD art by photographing it.
#Other countries are less merciful.
PD_ART_CATEGORY = "Category:PD Art"

#Uncopyrightable photograph of PD art that's 100 years old or older.
#Should exclude all the Italian photographs that are questionably PD in the US.
PD_ART_CATEGORY_OLD100 = "Category:PD-Art (PD-old-100)"

#Uncopyrightable photograph of art that has expired copyright in the US.
#Almost certainly the most safe category, and likely to be a high quality
#training source as it is art.
PD_ART_US_EXPIRATION_CATEGORY = "Category:PD-Art (PD-US-expired)"

def image_is_valid(file):
    """Test if an image file on disk loads with PIL."""

    try:
        test_image = Image.open(os.path.abspath(file))
        try:
            test_image.load()
            test_image.close()

            return True
        except PIL.UnidentifiedImageError:
            print ("Warning: Image {} is an unknown format".format(file))
            test_image.close()

            return False
    except PIL.Image.DecompressionBombError:
        print ("Warning: Image {} is too large for PIL".format(file))
        return False
    except OSError as e:
        print ("Warning: Image {} could not be read from disk, error was: {}".format(file, e))
        return False

def scrape_and_save_metadata(conn, localfile, item=None, rescrape=False):
    """Scrape data from the Wikimedia connection and item to the local file path given.
    
    Item must be a dict with "title" and "pageid" keys matching the item to
    download. Localfile must be a valid filesystem path. If not provided, we
    will attempt to recover it from the title in the stored metadata. If you
    are scraping from a Mediawiki server, the `walk_category` method of
    `Connection` will provide item dicts for you.
    
    Returns True if the file scraped successfully and new data was obtained.
    False indicates one of the following possible error conditions:
    
     - The file is already available locally and is up-to-date
     - The file cannot be saved because it exceeds PIL's size requirements
    
    If you are enforcing a scrape limit, be sure not to count failures against
    that limit."""
    metadatafile = localfile + '.json'

    if os.path.exists(localfile + ".banned") or os.path.exists(localfile + ".bannedmetadata"):
        #Skip downloaded files that were banned from the training set.
        #Images can be banned either because they were too large to decode,
        #or because the file could not be decoded in PIL.
        return False

    true_item_name = None
    true_pageid = None
    if item is not None:
        true_item_name = item["title"]
        true_pageid = item["pageid"]
    
    file_already_exists = os.path.exists(localfile)
    metadata_already_exists = os.path.exists(metadatafile)
    if metadata_already_exists:
        #Check if we need to re-scrape categories or wikitext
        with open(metadatafile, 'r') as source:
            source_data = json.load(source)

            if "categories" not in source_data or "parsetree" not in source_data:
                metadata_already_exists = False
            
            if "timestamp" not in source_data or "revisions" not in source_data:
                metadata_already_exists = False
            
            if "item" in source_data:
                true_item_name = source_data["item"]["title"]
                true_pageid = source_data["item"]["pageid"]
            else:
                #Attempt to recover the page name from the stripped titles we
                #saved in earlier runs. We only ever stored file metadata in
                #this state, so we can assume what we stored earlier was files.
                ext = os.path.splitext(localfile)[1]
                if not source_data["title"].endswith(ext):
                    true_item_name = f"File:{source_data['title']}{ext}"
                else:
                    #Some of our data has file extensions and some of it doesn't
                    #We should probably discontinue use of the stripped title field
                    true_item_name = f"File:{source_data['title']}"
                
                metadata_already_exists = False
    
    if true_item_name is None:
        raise Exception(f"Local metadata file {localfile} does not exist and there is no page title to work with")
    
    if true_pageid is None:
        #We can recover from this, too.
        #Just store the first page ID we see.
        pageinfo = conn.info(titles=[true_item_name])
        for pageid in pageinfo["query"]["pages"].keys():
            if true_pageid is not None:
                raise Exception(f"Local metadata file {localfile} does not have an unambiguous page ID")

            true_pageid = pageid
        
        if true_pageid is None:
            raise Exception(f"Local metadata file {localfile} has been deleted")
    
    if file_already_exists and metadata_already_exists:
        return False
    
    print(true_item_name)
    
    image_info = conn.image_info(titles=[true_item_name], iiprop=["url", "size"])["query"]["pages"][str(true_pageid)]["imageinfo"]
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
        metadata["item"] = {
            "title": true_item_name,
            "pageid": true_pageid
        }
        metadata["title"] = true_item_name.removeprefix("File:").removesuffix(".jpg").removesuffix(".jpeg").removesuffix(".png").removesuffix(".tif").removesuffix(".tiff")
        
        rv_timestamp = conn.revisions(titles=[true_item_name], rvprop=["timestamp"])
        if "revisions" in rv_timestamp["query"]["pages"][str(true_pageid)]:
            metadata["timestamp"] = rv_timestamp["query"]["pages"][str(true_pageid)]["revisions"][0]["timestamp"]

            revisions = rv_timestamp["query"]["pages"][str(true_pageid)]["revisions"]
            if "continue" in rv_timestamp:
                rvcontinue = rv_timestamp["continue"]["rvcontinue"]

                while rvcontinue is not None:
                    rvcontinue_data = conn.revisions(titles=[true_item_name], rvcontinue=rvcontinue)
                    if "revisions" in rvcontinue_data["query"]["pages"][str(true_pageid)]:
                        for rev in rvcontinue_data["query"]["pages"][str(true_pageid)]["revisions"]:
                            revisions.append(rev)
                    
                    if "continue" in rvcontinue_data:
                        rvcontinue = rvcontinue_data["continue"]["rvcontinue"]
                    else:
                        rvcontinue = None
            
            metadata["revisions"] = revisions

        page_terms = conn.page_terms(titles=[true_item_name])
        if "terms" in page_terms["query"]["pages"][str(true_pageid)]:
            metadata["terms"] = page_terms["query"]["pages"][str(true_pageid)]["terms"]
        
        page_cats = conn.categories(titles=[true_item_name])
        if "categories" in page_cats["query"]["pages"][str(true_pageid)]:
            metadata["categories"] = page_cats["query"]["pages"][str(true_pageid)]["categories"]
        
        page_text = conn.parse_tree(true_item_name)
        if "parsetree" in page_text["parse"]:
            metadata["parsetree"] = page_text["parse"]["parsetree"]
        
        metadata["imageinfo"] = image_info
        
        with open(metadatafile, 'w') as sink:
            sink.write(json.dumps(metadata))
    
    return not image_is_banned

def local_wikimedia(limit = None, prohibited_categories=[], load_images=True, intended_maximum_size=512):
    """Load in training data previously downloaded by running this module's main.
    
    Intended to be used as a Huggingface dataset via:
    
    ```from datasets import Dataset
    from PDDiffusion.datasets.WikimediaCommons import local_wikimedia
    
    data = Dataset.from_generator(local_wikimedia)```"""
    
    count = 0

    resize_cache = os.path.join(LOCAL_RESIZE_CACHE, str(intended_maximum_size))
    if not os.path.exists(resize_cache):
        os.makedirs(resize_cache)
    
    for file in itertools.chain(
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpg")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpeg")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.png")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.tif")),
            glob.iglob(os.path.join(LOCAL_STORAGE, "*.tiff"))
        ):
        if limit is not None and count > limit:
            return
        
        count += 1

        label = None
        if os.path.exists(file + ".json"):
            with open(file + ".json", "r") as metadata:
                metadata_obj = json.load(metadata)
                label = metadata_obj["title"]

                is_prohibited = False

                for category in metadata_obj["categories"]:
                    if category["title"] in prohibited_categories:
                        print("{} is prohibited".format(category["title"]))
                        is_prohibited = True
                        break
                
                if is_prohibited:
                    continue

                #TODO: Should we yield the same image twice with different data?
                try:
                    for extra in metadata_obj["terms"]["label"]:
                        label = label + ", " + extra
                except:
                    pass
                
                for name in metadata_obj["parsetree"]:
                    xmlstr = metadata_obj["parsetree"][name]

                    try:
                        extracted = extract_information_from_wikitext(xmlstr)

                        #We add in reverse order here to both:
                        #  - Avoid obliterating data extracted from the terms
                        #  - Keep the most concise/important data first since CLIP has
                        #    a maximum token count
                        if "description" in extracted:
                            label = extracted["description"] + " " + label
                        
                        if "medium" in extracted:
                            label = extracted["medium"] + " " + label
                        
                        if "title" in extracted:
                            label = extracted["title"] + " " + label

                        if "date" in extracted:
                            label = extracted["date"] + " " + label

                        if "artist" in extracted:
                            label = extracted["artist"] + " " + label

                        if "object type" in extracted:
                            label = extracted["object type"] + " " + label
                    except Exception as e:
                        pass
        
        if label is None:
            print ("Warning: Image {} is unlabeled, skipping".format(file))
            continue
        
        if load_images:
            #Check if our image has been resized down already, and if so use it.
            resized_file = os.path.join(resize_cache, os.path.basename(file))
            if os.path.exists(resized_file) and image_is_valid(resized_file):
                file = resized_file
            else:
                if not image_is_valid(file):
                    #Rename the file and its metadata for later inspection
                    os.rename(file, file + '.banned')
                    os.rename(file + '.json', file + '.bannedmetadata')

                    continue
                
                #Otherwise check if this image is too large and if so, downscale.
                image = Image.open(file)
                if image.width > intended_maximum_size or image.height > intended_maximum_size:
                    image.thumbnail((intended_maximum_size, intended_maximum_size))
                    image.save(resized_file)
                    image.close()

                    file = resized_file
                else:
                    image.close()
            
            image = Image.open(os.path.abspath(file))
            image.close()

            yield {"image": image, "image_file_path": os.path.abspath(file), "text": label}
        else:
            yield {"image_file_path": os.path.abspath(file), "text": label}
