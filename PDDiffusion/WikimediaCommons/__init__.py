import requests

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

PD_CATEGORY = "Category:Public domain"

PD_EXPIRATION_CATEGORY = "Category:Public_domain_due_to_copyright_expiration"