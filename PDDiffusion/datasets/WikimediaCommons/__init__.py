import requests, urllib.request, os.path, PIL, dateutil.parser, time
from PIL import Image
from PDDiffusion.datasets.WikimediaCommons.wikiparse import extract_information_from_wikitext
from PDDiffusion.datasets.WikimediaCommons.model import WikimediaCommonsImage, BASE_API_ENDPOINT
from PDDiffusion.datasets.model import DatasetImage, DatasetLabel, File
from PDDiffusion.datasets.validity import image_is_valid
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.engine import Row

import itertools

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

LOCAL_STORAGE = os.path.join("sets", "wikimedia")
LOCAL_RESIZE_CACHE = os.path.join("sets", "wikimedia-cache")

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
    
    def walk_category(self, category_name, member_types=["page"], recursive = True, visitedcats=None):
        """Walk a category on the given wiki API.
        
        Yields member pages or subcategories in the category as returned from
        the API. This includes a title and pageid at minimum.

        Member types lists all the types you care about.
        
        In Recursive mode, all subcategories will also be walked.
        
        `visitedcats` is internal state to prevent visiting the same category
        repeatedly. It must be populated with a set; any member of the set will
        be skipped when walking."""

        cmcontinue = None
        my_cmtype = set(member_types)
        if recursive:
            #We always need to ask for subcats in a recursive walk.
            my_cmtype.add("subcat")
        
        if visitedcats is None:
            visitedcats = set()
        
        while True:
            page = self.get(action="query", list="categorymembers", cmtitle=category_name, cmlimit=20, cmcontinue=cmcontinue, cmtype="|".join(my_cmtype))

            for item in page["query"]["categorymembers"]:
                if item["title"] in visitedcats:
                    continue

                visitedcats.add(item["title"])

                if recursive and item["title"].startswith("Category:"):
                    if "subcat" in member_types:
                        #If the user asked for both subcats AND a recursive walk,
                        #we have to yield both the category and its children
                        yield item
                    
                    yield from self.walk_category(item["title"], member_types=member_types, recursive=recursive, visitedcats=visitedcats)
                else:
                    yield item
            
            if "continue" in page and "cmcontinue" in page["continue"]:
                cmcontinue = page["continue"]["cmcontinue"]
            else:
                return
    
    def query(self, titles=[], prop=[], **kwargs):
        """Query for any number of properties in a given set of pages.

        Extra parameters for the query support list properties. Lists will be
        merged using | characters, which is MediaWiki standard.
        
        At a minimum, you must provide a list of page titles and a list of
        properties for those pages. The returned data will be structured as
        follows:
        
        ```
        {
            "batchcomplete": "", /* If all data is present */
            "continue": { /* If more requests are needed to get all data */
                "continue": "", /* Value of the continue property that will
                                 * get you the next bit of data */
                "clcontinue": "", /* Ibid but for clcontinue */
            }
            "query": {
                "pages": {
                    "12345": { /* All page specific properties go here. */ }
                    "67890": { /* Ibid. */ }
                }
            }
        }
        ```"""
        parsed_kwargs = {}
        for key in kwargs.keys():
            if type(kwargs[key]) == list:
                parsed_kwargs[key] = "|".join(kwargs[key])
            else:
                parsed_kwargs[key] = kwargs[key]
        
        return self.get(action="query", titles="|".join(titles), prop="|".join(prop), **parsed_kwargs)
    
    def query_all(self, titles=[], prop=[], **kwargs):
        """Query for any number of properties and keep querying until the page
        is out of data.
        
        We merge data from multiple queries as follows:
        
         * If a value is a string, the last seen value is recorded.
         * If a value is a list, we append onto that list.
         * If a value is a dict, we recursively apply the rules to its keys and
         values."""
        
        if len(titles) > 50:
            output = {}
            for chunk in chunked_iterable(titles, 50):
                output = dict_merge(output, self.query_all(titles=chunk, prop=prop, **kwargs))
            
            return output
        else:
            last_query = self.query(titles, prop, **kwargs)
            data = last_query["query"]
        
        while "batchcomplete" not in last_query:
            continue_kwargs = kwargs.copy()
            for continue_key in last_query["continue"].keys():
                continue_kwargs[continue_key] = last_query["continue"][continue_key]
            
            last_query = self.query(titles, prop, **continue_kwargs)
            data = dict_merge(data, last_query["query"], path=["query"])
        
        last_query["query"] = data
        return last_query
    
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

def dict_merge(dict1, dict2, path=[]):
    """Merge dictionaries' keys in such a way that lists are merged rather than
    being overwritten.
    
    This mutates the first dict, like dict.update.
    
    `path` is the full path to the data being merged from the root object.
    Certain paths are special-cased (e.g. query pages) to deal with Mediawiki
    API weirdness, such as negative IDs being assigned ad-hoc to pages that
    don't exist. Negative IDs in dict2 with conflicting ID numbers to that of
    dict1 will be renumbered in this case, with page titles being used to
    detect conflicts."""
    
    pages_to_ids = {}
    if path == ["query", "pages"]:
        last_denormal_key = 0
        for key in dict1.keys():
            pages_to_ids[dict1[key]["title"]] = key
            if int(key) < 0:
                last_denormal_key = min(last_denormal_key, int(key))
        
        for key in dict2.keys():
            if dict2[key]["title"] not in pages_to_ids:
                if int(key) > 0:
                    pages_to_ids[dict2[key]["title"]] = key
                else:
                    last_denormal_key -= 1
                    if last_denormal_key in dict1:
                        raise Exception(f"Key {last_denormal_key} should be unique but isn't")

                    pages_to_ids[dict2[key]["title"]] = str(last_denormal_key)
        
        for key in dict2.keys():
            normalized_key = pages_to_ids[dict2[key]["title"]]
            if normalized_key not in dict1:
                dict1[normalized_key] = dict2[key]
            elif type(dict1[normalized_key]) == list and type(dict2[key]) == list:
                dict1[normalized_key] = dict1[normalized_key] + dict2[key]
            elif type(dict1[normalized_key]) == dict and type(dict2[key]) == dict:
                dict1[normalized_key] = dict_merge(dict1[normalized_key], dict2[key], path=path + [key])
            else:
                if type(dict1[normalized_key]) != type(dict2[key]):
                    raise Exception(f"Type of key {key} changed from {type(dict1[normalized_key])} to {type(dict2[key])} across requests")
                
                dict1[normalized_key] = dict2[key]
    else:
        for key in dict2.keys():
            if key not in dict1:
                dict1[key] = dict2[key]
            elif type(dict1[key]) == list and type(dict2[key]) == list:
                dict1[key] = dict1[key] + dict2[key]
            elif type(dict1[key]) == dict and type(dict2[key]) == dict:
                dict1[key] = dict_merge(dict1[key], dict2[key], path=path + [key])
            else:
                if type(dict1[key]) != type(dict2[key]):
                    raise Exception(f"Type of key {key} changed from {type(dict1[key])} to {type(dict2[key])} across requests")
                
                dict1[key] = dict2[key]
    
    return dict1

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

def extract_labels_for_article(session, article):
    """Extract labels from a Wikimedia Commons article and store them in its associated image."""

    for label in article.base_image.labels:
        session.delete(label)
    
    session.flush()

    extracts = {}
    if "terms" in article.wikidata:
        for key in article.wikidata["terms"].keys():
            extracts[f"__{key}"] = ", ".join(article.wikidata["terms"][key])

    for name in article.wikidata["parsetree"]:
        xmlstr = article.wikidata["parsetree"][name]
        try:
            extracts.update(extract_information_from_wikitext(xmlstr))
        except Exception as e:
            print(f"Got error when extracting information from wikitext: {e}")
    
    for key in extracts.keys():
        label = DatasetLabel(image_id=article.id, dataset_id=article.dataset_id, data_key=key, value=extracts[key])
        
        article.base_image.labels.append(label)
        session.add(label)

def transient_error_tolerance(fn):
    def ret(conn, session, *args, **kwargs):
        tries = 0

        while True:
            try:
                return fn(conn, session, *args, **kwargs)
            except urllib.request.HTTPError as e:
                if e.code == 502 and tries < 5:
                    session.rollback()
                    
                    tries += 1
                    time.sleep(tries)
                    print(f"Retrying prior scrape ({tries} tries)")
                else:
                    raise e
    
    return ret

@transient_error_tolerance
def scrape_and_save_metadata(conn, session, pages=[], force_rescrape=False):
    """Scrape data from the Wikimedia connection and item to the local file path given.

    Conn is the current Mediawiki API connection and session is the current
    database connection.

    Pages is a list of pages to query. Pages can be provided either as an item
    dict (w/"title" and "pageid" keys), a pair of database items
    (article, image), or a bare page title string. Pages will be deduplicated
    by Python equality before use.
    
    If you are scraping from a Mediawiki server, the `walk_category` method of
    `Connection` will provide item dicts for you.

    Returns the number of images that were successfully scraped. Data will not
    be scraped in the following error conditions:
    
     - The file is already available locally and is up-to-date
     - The file cannot be saved because it exceeds PIL's size requirements"""
    
    dataset_id = f"WikimediaCommons:{conn.base_api_endpoint}"

    items = {}               #List of pages we need to scrape
    titles_to_query = set()  #List of page titles to put into the metadata query
    successful_items = set() #List of pages that we scraped successfully, for limit counting

    #If we have bare titles we haven't seen before, we have to ask for the IDs
    unknown_id_titles = []
    unknown_id_page_ids = {}
    for page in pages:
        if type(page) == str:
            localdata = session.execute(
                select(WikimediaCommonsImage, DatasetImage)
                    .outerjoin(DatasetImage, WikimediaCommonsImage.id == DatasetImage.id)
                    .where(WikimediaCommonsImage.id == page, WikimediaCommonsImage.dataset_id == dataset_id)
            ).one_or_none()

            if localdata is None:
                unknown_id_titles.append(page)
            else:
                (article, image) = localdata
                if article.post_id > 0:
                    unknown_id_page_ids[article.id] = article.post_id
                else:
                    unknown_id_titles.append(article.id)
    
    #MediaWiki can arbitrarily decide to 'normalize' page titles, and we have
    #to look things up on our end if such a thing has happened
    to_normalized_title = {}

    if len(unknown_id_titles) > 0:
        the_ids_query = conn.query_all(titles=unknown_id_titles)
        if "normalized" in the_ids_query["query"]:
            for normalization in the_ids_query["query"]["normalized"]:
                to_normalized_title[normalization["from"]] = normalization["to"]

        for page_id in the_ids_query["query"]["pages"].keys():
            corresponding_title = the_ids_query["query"]["pages"][page_id]["title"]
            if int(page_id) >= 0:
                unknown_id_page_ids[corresponding_title] = page_id
            else:
                unknown_id_page_ids[corresponding_title] = "-1"
    
    for page in pages:
        item = {}

        #Bare titles get turned into dicts
        if type(page) == str:
            page = {"title": page}
        
        #Then, we need to determine if we've gotten a page title/id pair, or a
        #database item.
        if type(page) == dict: #pageid/title dict
            if page["title"] in to_normalized_title:
                print(f"Normalized {page['title']} to {to_normalized_title[page['title']]}")
                page["title"] = to_normalized_title[page["title"]]
            
            localdata = session.execute(
                select(WikimediaCommonsImage, DatasetImage)
                    .outerjoin(DatasetImage, WikimediaCommonsImage.id == DatasetImage.id)
                    .where(WikimediaCommonsImage.id == page["title"], WikimediaCommonsImage.dataset_id == dataset_id)
            ).one_or_none()

            if localdata is not None:
                (article, image) = localdata
                page["pageid"] = article.post_id
            else:
                if "pageid" not in page:
                    if page["title"] not in unknown_id_page_ids:
                        print(f"{page['title']} has no page ID, skipping")
                        continue

                    page["pageid"] = unknown_id_page_ids[page["title"]]
                
                article = WikimediaCommonsImage(dataset_id=dataset_id, id=page["title"], post_id=page["pageid"])

                if page["title"].startswith("File:"):
                    image = DatasetImage(dataset_id=dataset_id, id=page["title"])
                    article.base_image = image
                else: #Category or other non-image file that we want to retain anyway
                    image = None
                
                session.add(article)

            item = {
                "title": page["title"],
                "pageid": page["pageid"],
                "article": article,
                "image": image
            }
        elif type(page) == tuple or type(page) == Row: #article/image orm object pair
            (article, image) = page

            item = {
                "title": article.id,
                "pageid": article.post_id,
                "article": article,
                "image": image
            }
        
        #Next, we need to decide what to scrape.
        file_already_exists = image is None #No image means we don't care to scrape it.
        metadata_already_exists = False
        
        if image is not None and image.file is not None:
            file_already_exists = True

            #Check if the image is actually stored or if we just have a bogus
            #file entry.
            if image.file.storage_provider != File.LOCAL_FILE:
                print(f"Non-local file provider {image.file.storage_provider}")

            if not os.path.exists(image.file.url):
                file_already_exists = False
                image.is_banned = False
        
        if article.wikidata is not None:
            metadata_already_exists = True

            # Check if any bits of data are missing from our Wiki scrape.
            # If so, force a redownload.
            if "categories" not in article.wikidata or "parsetree" not in article.wikidata:
                metadata_already_exists = False
            
            if "timestamp" not in article.wikidata or "revisions" not in article.wikidata:
                metadata_already_exists = False
            
            if "revisions" in article.wikidata and len(article.wikidata["revisions"]) > 0 and "user" not in article.wikidata["revisions"][0]:
                metadata_already_exists = False
            
            if "hidden_categories" not in article.wikidata:
                metadata_already_exists = False
        
        if article.last_edited is None:
            metadata_already_exists = False

        if image is not None and image.is_banned:
            #Skip downloaded files that were banned from the training set.
            #Images can be banned either because they were too large to decode,
            #or because the file could not be decoded in PIL.
            continue

        if force_rescrape:
            metadata_already_exists = False
        
        #TODO: Handle redirects.
        #File:Gérard - Eugène de Beauharnais 1.jpg is an example redirect
    
        if file_already_exists and metadata_already_exists:
            continue

        print(item["title"])
        
        if not metadata_already_exists:
            titles_to_query.add(item["title"])
        
        item["file_already_exists"] = file_already_exists
        items[item["title"]] = item
    
    cats_to_query = set()
    
    if len(titles_to_query) > 0:
        #The actual query. This is done once with the list of things we want to
        #query for fewer HTTP requests
        query_data = conn.query_all(
            titles=list(titles_to_query),
            prop=["imageinfo", "revisions", "pageterms", "categories"],
            iiprop=["url", "size"],
            iilimit=100,
            cllimit="max",
            clshow="!hidden",
            rvprop=["timestamp", "user"]
        )

        hidden_category_data = conn.query_all(
            titles=list(titles_to_query),
            prop=["categories"],
            clshow="hidden"
        )

        titles_returned = set()
        
        #All the pages we query now get copied back into their respective objects.
        for page_id in query_data["query"]["pages"].keys():
            corresponding_title = query_data["query"]["pages"][page_id]["title"]

            if int(page_id) > 0 and items[corresponding_title]["pageid"] != page_id:
                #Sometimes pages get renumbered, so we need to handle that.
                #Page ID of -1 means "no page"
                items[corresponding_title]["pageid"] = page_id
                items[corresponding_title]["article"].post_id = page_id

                if items[corresponding_title]["article"].wikidata is not None:
                    if "item" not in items[corresponding_title]["article"].wikidata:
                        items[corresponding_title]["article"].wikidata["item"] = {}
                    else: #paranoia/sanity check
                        if items[corresponding_title]["article"].wikidata["item"]["title"] != corresponding_title:
                            raise Exception(f"Item {items[corresponding_title]['article'].wikidata['item']['title']} got renumbered but also renamed to {corresponding_title}")
                    
                    items[corresponding_title]["article"].wikidata["item"]["title"] = corresponding_title
                    items[corresponding_title]["article"].wikidata["item"]["pageid"] = page_id
            
            metadata = {}
            metadata["item"] = {
                "title": corresponding_title,
                "pageid": page_id
            }
            
            page_text = conn.parse_tree(corresponding_title)
            if "parse" in page_text and "parsetree" in page_text["parse"]:
                metadata["parsetree"] = page_text["parse"]["parsetree"]
            elif "error" in page_text:
                print(f"{corresponding_title} Error: {page_text['error']['info']}")

                #Don't import the page, just delete it
                if page_text["error"]["info"] == "The page you specified doesn't exist.":
                    titles_returned.add(corresponding_title)
                    #TODO: Delete when SQLAlchemy won't shit itself
                    
                    continue

            #Legacy title, probably should be removed.
            metadata["title"] = corresponding_title.removeprefix("File:").removesuffix(".jpg").removesuffix(".jpeg").removesuffix(".png").removesuffix(".tif").removesuffix(".tiff")
            
            if "revisions" in query_data["query"]["pages"][page_id]:
                metadata["timestamp"] = query_data["query"]["pages"][page_id]["revisions"][0]["timestamp"]
                items[corresponding_title]["article"].last_edited = dateutil.parser.isoparse(metadata["timestamp"])

                revisions = query_data["query"]["pages"][page_id]["revisions"]
                metadata["revisions"] = revisions
            
            if "terms" in query_data["query"]["pages"][page_id]:
                metadata["terms"] = query_data["query"]["pages"][page_id]["terms"]
            
            all_cats = []
            
            if "categories" in query_data["query"]["pages"][page_id]:
                all_cats += query_data["query"]["pages"][page_id]["categories"]
            
            if "categories" in hidden_category_data["query"]["pages"][page_id]:
                all_cats += hidden_category_data["query"]["pages"][page_id]["categories"]
            
            metadata["categories"] = all_cats
            metadata["hidden_categories"] = True
            
            for catref in all_cats:
                cats_to_query.add(catref["title"])
            
            if "imageinfo" in query_data["query"]["pages"][page_id]:
                metadata["imageinfo"] = query_data["query"]["pages"][page_id]["imageinfo"]
            
            items[corresponding_title]["article"].wikidata = metadata

            if corresponding_title.startswith("File:"):
                extract_labels_for_article(session, items[corresponding_title]["article"])

            titles_returned.add(corresponding_title)
            successful_items.add(corresponding_title)
        
        #More paranoia: we should always get the same amount of titles in and out
        if len(titles_returned) != len(titles_to_query):
            raise Exception(f"Query imbalance: asked for info on {len(titles_to_query)} pages but got {len(titles_returned)} back")
    
    #Now check if we have any images to scrape.
    for corresponding_title in items.keys():
        if not items[corresponding_title]["file_already_exists"]:
            image_info = items[corresponding_title]["article"].wikidata["imageinfo"]
            if image_info[0]["size"] > Image.MAX_IMAGE_PIXELS:
                #Don't even download the file, just mark the metadata as banned
                items[corresponding_title]["article"].is_banned = True
                continue

            localfile = corresponding_title.removeprefix("File:").replace("\"", "").replace("'", "").replace("?", "").replace("!", "").replace("*", "").strip()
            localfile = os.path.join(LOCAL_STORAGE, localfile)

            #NOTE: This is intended to deal with an old version of the code that
            #didn't commit between each download, so if you Ctrl-C'd it you lost
            #all the metadata. If we already have the file, just don't redownload
            #it.
            #
            #When we support image revision redownloading we need to check if the
            #image was updated and OVERRIDE this check.
            if not os.path.exists(localfile):
                with conn.urlopen(image_info[0]["url"]) as source:
                    with open(localfile, "wb") as sink:
                        sink.write(source.read())
            
            items[corresponding_title]["image"].file = File(storage_provider=File.LOCAL_FILE, url=localfile)
            session.add(items[corresponding_title]["image"].file)
            
            if not image_is_valid(localfile):
                items[corresponding_title]["image"].is_banned = True
            
            successful_items.add(corresponding_title)

    return len(successful_items)

def local_wikimedia_base(limit = None, prohibited_categories=[], load_images=True, intended_maximum_size=512):
    """Load in training data previously downloaded by running this module's main.
    
    The yielded data items will be minimally processed: that is, all text fields
    will be returned separately instead of being merged together for ML label
    processing.
    
    If load_images is enabled, PIL image data will be provided. Images will be
    shrunk to fit the intended_maximum_size and stored on disk."""

    resize_cache = os.path.join(LOCAL_RESIZE_CACHE, str(intended_maximum_size))
    if not os.path.exists(resize_cache):
        os.makedirs(resize_cache)
    
    load_dotenv()
    engine = create_engine(os.getenv("DATABASE_CONNECTION"), future=True)

    with Session(engine) as session:
        #Datasets will absolutely not tolerate any raggedness in items, and even a
        #defaultdict fails to correctly fix the problem. We have to grab all the
        #keys first so we know to add empty strings to things
        keys = set()
        for (key,) in session.execute(select(DatasetLabel.data_key).group_by(DatasetLabel.data_key)):
            keys.add(key)

        count = 0

        for (article, image) in WikimediaCommonsImage.select_all_image_articles(session):
            if limit is not None and count > limit:
                break
            
            count += 1

            extracted = {}
            metadata_obj = article.wikidata

            is_prohibited = False

            if "categories" in metadata_obj:
                for category in metadata_obj["categories"]:
                    if category["title"] in prohibited_categories:
                        print("{} is prohibited".format(category["title"]))
                        is_prohibited = True
                        break
            else:
                #TODO: When scraping we should ensure "no categories" is represented as an empty list
                print(f"{metadata_obj['item']['title']} has no categories")
            
            if is_prohibited:
                continue

            extracted["__pagetitle"] = metadata_obj["item"]["title"]
            extracted["__pageid"] = str(metadata_obj["item"]["pageid"])

            if image.file is None:
                continue

            if image.is_banned:
                continue

            if image.file.storage_provider != File.LOCAL_FILE:
                print(f"Non-local file provider {image.file.storage_provider}")

            file = image.file.url
            extracted["image_file_path"] = os.path.abspath(file)

            for label in image.labels:
                extracted[label.data_key] = label.value
            
            if load_images:
                #Check if our image has been resized down already, and if so use it.
                resized_file = os.path.join(resize_cache, os.path.basename(file))
                if os.path.exists(resized_file) and image_is_valid(resized_file):
                    file = resized_file
                else:
                    print(file)

                    if not image_is_valid(file):
                        image.is_banned = True
                        session.commit()

                        #Rename the file and its metadata for later inspection
                        try:
                            os.rename(file, file + '.banned')
                        except OSError:
                            pass

                        continue

                    #Otherwise check if this image is too large and if so, downscale.
                    try:
                        image = Image.open(file)
                        if image.width > intended_maximum_size or image.height > intended_maximum_size:
                            image.thumbnail((intended_maximum_size, intended_maximum_size))
                            image.save(resized_file)
                            image.close()

                            file = resized_file
                        else:
                            image.close()
                    except:
                        image.is_banned = True
                        session.commit()
                
                image = Image.open(os.path.abspath(file))
                image.close()

                extracted["image"] = image
            
            for key in keys:
                if key not in extracted:
                    extracted[key] = ""
            
            yield extracted

def local_wikimedia(limit = None, prohibited_categories=[], load_images=True, intended_maximum_size=512):
    """Load in training data previously downloaded by running this module's main.

    Synthesizes a label field from extracted wikitext in the process. This
    function is deprecated, callers should migrate to local_wikimedia_base and
    then decide how they want to handle the Wikimedia Commons label fields. For
    example, if you are training a text tokenizer, you may want to process all
    fields; but if you're training CLIP you would want to change the fields out
    every epoch as a data augmentation strategy.
    
    Intended to be used as a Huggingface dataset via:
    
    ```from datasets import Dataset
    from PDDiffusion.datasets.WikimediaCommons import local_wikimedia
    
    data = Dataset.from_generator(local_wikimedia)```"""
    
    for item in local_wikimedia_base(
            limit=limit,
            prohibited_categories=prohibited_categories,
            load_images=load_images,
            intended_maximum_size=intended_maximum_size):
        
        #Synthesize a label from the item data

        label = item["__pagetitle"]

        if "__label" in item:
            label = label + ", " + item["__label"]

        #We add in reverse order here to both:
        #  - Avoid obliterating data extracted from the terms
        #  - Keep the most concise/important data first since CLIP has
        #    a maximum token count
        if "description" in item:
            label = item["description"] + " " + label
        
        if "medium" in item:
            label = item["medium"] + " " + label
        
        if "title" in item:
            label = item["title"] + " " + label

        if "date" in item:
            label = item["date"] + " " + label

        if "artist" in item:
            label = item["artist"] + " " + label

        if "object type" in item:
            label = item["object type"] + " " + label
        
        item["label"] = label

        yield item
