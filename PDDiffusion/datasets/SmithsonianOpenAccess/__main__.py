import json, os.path, re, datetime

S3_BUCKET_NAME = "smithsonian-open-access"
S3_BASE_PATH = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com/"
MASTER_INDEX = "metadata/edan/index.txt"

def read_index(index_file):
    """Read an index file and list all of the child index files.
    
    The format of an index file is a list of S3 URLs. They will be reformatted
    as OS paths before returning."""
    child_files = []

    for line in index_file:
        child_files.append(os.path.join(*line.strip().removeprefix(S3_BASE_PATH).split("/")))
    
    return child_files

def read_metadata(md_file, filter_func = None):
    """Read a metadata file and return all of the JSON objects.
    
    The format of a metadata file is multiple JSON objects, one per line.
    Effectively an implicit JSON array.
    
    If this function throws an exception, this is not a valid metadata file.
    
    The filter function can be used to exclude objects we aren't interested in
    from the result."""
    metadata_objects = []

    for line in md_file:
        obj = json.loads(line)
        
        if filter_func is not None and not filter_func(obj):
            continue

        metadata_objects.append(obj)
    
    return metadata_objects

def read_index_or_metadata(any_file, base_path, filter_func = None):
    """Read an index or metadata file.
    
    Returned data will be in one of two formats:
    
     * A list of metadata objects, if the file was metadata
     * A dictionary mapping file paths to their decoded values, if the file was
       an index. Values are the result of this function applied to that file.
       This forms a tree structure.
    
    The base path is the location of the local cache of Smithsonian metadata.
    It should have been synced with AWS CLI before running this code.
    
    The filter function allows removing objects from the result that we are not
    interested in."""
    
    try:
        return read_metadata(any_file, filter_func)
    except json.JSONDecodeError:
        pass

    child_objects = dict()

    for filename in read_index(any_file):
        with open(os.path.join(base_path, filename), 'r', encoding="utf-8") as child:
            read_index_or_metadata(child, base_path, filter_func)
            #TODO: Actually storing the child objects uses too much RAM
            #child_objects[filename] = read_index_or_metadata(child, base_path, filter_func)
    
    return child_objects

def select_images(object):
    """Filter that selects for media that has an associated image."""
    if "descriptiveNonRepeating" in object["content"] \
        and "online_media" in object["content"]["descriptiveNonRepeating"] \
        and "media" in object["content"]["descriptiveNonRepeating"]["online_media"] \
        and len(object["content"]["descriptiveNonRepeating"]["online_media"]["media"]) > 0:
        return True
    else:
        return False

WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR = 1923

def select_public_domain_expired(object):
    """Filter that selects for collection objects that are public domain
    worldwide due to the work's copyright having expired. We use the cutoff date
    of January 1st, 1923 as that is the cutoff date according to US law.

    The date metadata in the Smithsonian dataset is weird: dates are provided as
    date ranges (e.g. "1880s") in some cases. We are conservative and only pick
    records that do not mention any date beyond 1923.
    
    This function must be updated as international copyright laws change."""

    if "indexedStructured" in object["content"] \
        and "date" in object["content"]["indexedStructured"]:

        # No date data is legally risky.
        if len(object["content"]["indexedStructured"]["date"]) == 0:
            return False
        
        # Some works have multiple dates, we reject anything that is beyond the
        # cutoff.
        for date in object["content"]["indexedStructured"]["date"]:
            first_date = None
            cutoff = WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR

            if date.startswith("BCE ") and date.endswith("s"):
                first_date = int(date.removeprefix("BCE ").removesuffix("s")) * -1 + 9
                cutoff = WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR - (WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR % 10)
            elif date.startswith("BCE "):
                first_date = int(date.removeprefix("BCE ")) * -1
                cutoff = WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR
            elif date.endswith("s"):
                first_date = int(date.removesuffix("s")) + 9 #Most pessimistic possible interpretation
                cutoff = WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR - (WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR % 10)
            
            if first_date is None:
                #Unparseable date, error out
                print(object["id"])
                print(date)
                return False
            
            if first_date >= cutoff:
                return False
        
        # At least one date was provided and all dates are public domain.
        return True
    
    if "freetext" in object["content"] \
        and "date" in object["content"]["freetext"]:

        # No date data is legally risky.
        if len(object["content"]["freetext"]["date"]) == 0:
            return False
        
        for date in object["content"]["freetext"]["date"]:
            first_date = None
            cutoff = WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR

            if date["content"] == "unknown" or date["content"] == "n.d." or date["content"] == "n.d" \
                or date["content"] == "n.d.a." or date["content"] == "Date of issue: unknown":
                #Unknown date is legally risky.
                return False
            
            if date["content"].startswith("after ") or date["content"].startswith("After "):
                #No upper bound on date is legally risky.
                return False
            
            if date["content"].lower() == "prehistoric":
                #We can't get sued by cavemen, *can we*?
                return True
            
            if date["content"].endswith("nineteenth century") \
                or date["content"].endswith("19th C."):
                first_date = 1899
                cutoff = cutoff - (cutoff % 100)
            elif date["content"].endswith("eighteenth century"):
                first_date = 1799
                cutoff = cutoff - (cutoff % 100)
            elif date["content"] == "Circa World War II":
                first_date = 1945 #VJ/Victory over Japan day
            elif date["content"] == "Circa World War I":
                first_date = 1919 #Signing of the Treaty of Versailles
            elif date["content"].lower().endswith("st century") \
                or date["content"].lower().endswith("st century ad") \
                or date["content"].lower().endswith("st century a.d.") \
                or date["content"].lower().endswith("nd century") \
                or date["content"].lower().endswith("nd century ad") \
                or date["content"].lower().endswith("nd century a.d.") \
                or date["content"].lower().endswith("rd century") \
                or date["content"].lower().endswith("rd century ad") \
                or date["content"].lower().endswith("rd century a.d.") \
                or date["content"].lower().endswith("th century") \
                or date["content"].lower().endswith("th century ad") \
                or date["content"].lower().endswith("th century a.d."):
                #xxth-xxth century (AD) or xxth century (AD) or mid-xxth century
                #We only care about the end range
                parsed = re.search("(\d)$", date["content"].lower().removesuffix(" ad").removesuffix(" a.d.").removesuffix(" century").removesuffix("st").removesuffix("nd").removesuffix("rd").removesuffix("th"))

                if parsed is not None:
                    #Century numbers start from 1 because people don't believe "year zero" or "century zero" exists.
                    first_date = int(parsed.group(0)) * 100 - 100 + 99
                    cutoff = cutoff - (cutoff % 100)
            elif date["content"].endswith("st century BC") \
                or date["content"].endswith("nd century BC") \
                or date["content"].endswith("rd century BC") \
                or date["content"].endswith("th century BC") \
                or date["content"].endswith("st century B.C.") \
                or date["content"].endswith("nd century B.C.") \
                or date["content"].endswith("rd century B.C.") \
                or date["content"].endswith("th century B.C."):
                #xxth century BC
                #God hope this code is never legally necessary.
                parsed = re.search("(\d)$", date["content"].removesuffix(" B.C.").removesuffix(" BC").removesuffix(" century").removesuffix("st").removesuffix("nd").removesuffix("rd").removesuffix("th"))

                if parsed is not None:
                    first_date = int(parsed.group(0)) * -100 + 100 + 99
                    cutoff = cutoff - (cutoff % 100)
            elif date["content"].endswith(" AD") or date["content"].endswith(" A.D."):
                #xx BC-xx AD, or just xx AD
                #Dates are specified as full years but all examples I see of this are rounded to centuries.
                parsed = re.search("(\d)$", date["content"].removesuffix(" AD").removesuffix(" A.D."))

                if parsed is not None:
                    first_date = int(parsed.group(0))
                    cutoff = cutoff - (cutoff % 100)
            elif date["content"].endswith(" BC") or date["content"].endswith(" B.C."):
                #xx BC
                #God hope this code is never legally necessary.
                parsed = re.search("(\d)$", date["content"].removesuffix(" BC").removesuffix(" B.C."))

                if parsed is not None:
                    first_date = int(parsed.group(0)) * -1
                    cutoff = cutoff - (cutoff % 100)
            elif date["content"].startswith("c.") or date["content"].startswith("ca") and date["content"].endswith(" BCE"):
                #ca. xxx BC
                parsed = re.search("(\d)$", date["content"].removesuffix(" BCE"))

                if parsed is not None:
                    first_date = int(parsed.group(0)) * -1
                    cutoff = cutoff - (cutoff % 100)
            elif date["content"].startswith("c") or date["content"].startswith("C"):
                if date["content"].endswith("s"):
                    #ca/c/Circa 19xxs
                    parsed = re.search("(\d)$", date["content"].removesuffix("s"))

                    if parsed is not None:
                        first_date = int(parsed.group(0)) + 9 #Most pessimistic possible interpretation
                        cutoff = WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR - (WORLDWIDE_PUBLIC_DOMAIN_CUTOFF_YEAR % 10)
                else:
                    #ca. xxx, cxxx, Circa xxx, and other variants
                    #I'm guessing these are "circa this century"
                    parsed = re.search("(\d)$", date["content"])

                    if parsed is not None:
                        first_date = int(parsed.group(0))
                        cutoff = cutoff - (cutoff % 100)
            elif '-' in date["content"] or '–' in date["content"]:
                #xxx-xxx or xxx–xxx (that's an EM dash)
                #Some of these are precisely dated, so no cutoff adjustment.
                parsed = re.search("(\d)$", date["content"])

                if parsed is not None:
                    first_date = int(parsed.group(0)) * -1
            else:
                try:
                    first_date = datetime.datetime.strptime(date["content"].removeprefix("Printed ").removeprefix("Written ").replace("Februrary", "February").replace("Feburary", "February"), "%B %d, %Y").year
                except:
                    pass

                try:
                    first_date = datetime.datetime.strptime(date["content"][0:10], "%Y:%m:%d").year
                except:
                    pass

                try:
                    first_date = datetime.datetime.strptime(date["content"][0:10], "%m/%d/%Y").year
                except:
                    pass

                try:
                    first_date = int(date["content"])
                except:
                    pass
            
            if first_date is None:
                #Unparseable date, error out
                print(object["id"])
                print(date["content"])
                return False
            
            if first_date >= cutoff:
                return False
        
        # AT least one date was provided, we could understand it, and all dates are public domain.
        return True
    
    # No date data is legally risky.
    return False

def __main__():
    base_path = os.path.join("sets", "smithsonian")

    with open(os.path.join(base_path, MASTER_INDEX), 'r', encoding="utf-8") as parent:
        index = read_index_or_metadata(parent, base_path,
            filter_func = lambda obj: select_images(obj) and select_public_domain_expired(obj))

if __name__ == "__main__":
    __main__()
