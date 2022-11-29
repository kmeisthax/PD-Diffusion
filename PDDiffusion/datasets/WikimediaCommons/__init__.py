import requests, urllib.request, glob, os.path, itertools, json, PIL
from PIL import Image

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
    
    def image_info(self, titles=[], iiprop=[]):
        return self.get(action="query", prop="imageinfo", titles="|".join(titles), iiprop="|".join(iiprop))
    
    def page_terms(self, titles=[]):
        return self.get(action="query", prop="pageterms", titles="|".join(titles))
    
    def categories(self, titles=[]):
        """Get the categories for a given page."""
        return self.get(action="query", prop="categories", titles="|".join(titles), cllimit="max", clshow="hidden")
    
    def parse_tree(self, title):
        return self.get(action="parse", prop="parsetree", page=title)
    
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

def local_wikimedia(limit = None, prohibited_categories=["Category:Extracted images", "Category:Cropped images", "Category:Extracted drawings"], load_images=True, intended_maximum_size=512):
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
        
        if label is None:
            print ("Warning: Image {} is unlabeled, skipping".format(file))
            continue
        
        if load_images:
            if not image_is_valid(file):
                #Rename the file and its metadata for later inspection
                os.rename(file, file + '.banned')
                os.rename(file + '.json', file + '.bannedmetadata')

                continue

            #Check if our image has been resized down already, and if so use it.
            resized_file = os.path.join(resize_cache, os.path.basename(file))
            if os.path.exists(resized_file) and image_is_valid(resized_file):
                file = resized_file
            else:
                #Otherwise check if this image is too large and if so, downscale.
                image = Image.open(file)
                if image.width > intended_maximum_size or image.height > intended_maximum_size:
                    image.thumbnail((intended_maximum_size, intended_maximum_size))
                    image.save(resized_file)
                    image.close()

                    file = resized_file
            
            yield {"image": Image.open(os.path.abspath(file)), "image_file_path": os.path.abspath(file), "text": label}
        else:
            yield {"text": label}
