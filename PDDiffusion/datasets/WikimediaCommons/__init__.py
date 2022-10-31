import requests, urllib.request, glob, os.path, itertools, json, PIL
from PIL import Image

LOCAL_STORAGE = os.path.join("sets", "wikimedia")
BASE_API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"

DEFAULT_UA = "PD-Diffusion/0.0"

class Connection(object):
    def __init__(self, base_api_endpoint=BASE_API_ENDPOINT, ua=DEFAULT_UA):
        self.base_api_endpoint = base_api_endpoint
        self.ua = ua
    
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

def local_wikimedia(limit = None):
    """Load in training data previously downloaded by running this module's main.
    
    Intended to be used as a Huggingface dataset via:
    
    ```from datasets import Dataset
    from PDDiffusion.datasets.WikimediaCommons import local_wikimedia
    
    data = Dataset.from_generator(local_wikimedia)```"""
    
    count = 0
    
    for file in itertools.chain(glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpg")), glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpeg")), glob.iglob(os.path.join(LOCAL_STORAGE, "*.png"))):
        if limit is not None and count > limit:
            return
        
        count += 1

        label = os.path.splitext(os.path.basename(file))[0]
        if os.path.exists(file + ".json"):
            with open(file + ".json", "r") as metadata:
                #TODO: Should we yield the same image twice with different data?
                try:
                    for extra in json.load(metadata)["label"]:
                        label = label + ", " + extra
                except:
                    continue
        
        try:
            yield {"image": Image.open(os.path.abspath(file)), "image_file_path": os.path.abspath(file), "text": label}
        except PIL.UnidentifiedImageError:
            continue
