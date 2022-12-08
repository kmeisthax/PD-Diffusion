from defusedxml import ElementTree

def extract_languages_from_wikinode(wikinode):
    """Given a node in a parsed wikitext XML file, find all language template
    invocations in the value.
    
    This does not consider nested template invocations or templates buried in
    other styling."""

    langs = {}

    for tmpl in wikinode.findall("template"):
        lang = tmpl.find("title").text.strip()

        #We currently assume all two-letter templates are language codes.
        if len(lang) != 2:
            continue

        langs[lang] = []

        for part in tmpl.findall("part"):
            langs[lang].append(part.find("value"))
    
    return langs

def extract_text_from_language_value(wikinodes, warn=False):
    """Given a list of language values, extract all relevant text.
    
    See `extract_languages_from_wikinode for how to select one language from an
    Information template value. Alternatively, if your template value does not
    have language templates in it, you may provide the whole value wrapped in a
    one element list."""
    true_value = ""

    for lang_value in wikinodes:
        if lang_value.text:
            true_value = (true_value + " " + lang_value.text).strip()

        # Then check for other template types.
        for subtmpl in lang_value:
            if subtmpl.tag == "template":
                subtmpl_title = subtmpl.find("title").text.strip()

                #Assume that all creator templates do not have data values.
                if subtmpl_title.lower().startswith("creator:"):
                    true_value = (true_value + " " + subtmpl_title.removeprefix("Creator:").removeprefix("creator:")).strip()

                    if len(subtmpl.findall("part")) > 0:
                        if warn:
                            print(f"Creator template {true_value} has unknown data")
                else:
                    if warn:
                        print(f"Unknown value template {subtmpl_title}")
            else:
                if warn:
                    print(f"Unknown value tag {subtmpl.tag}")
            
            if subtmpl.tail:
                true_value = (true_value + " " + subtmpl.tail).strip()
    
    return true_value.strip()


def extract_information_from_wikitext(wikixml, preferred_lang = "en"):
    """Extract info from a Mediawiki XML parse tree related to a Wikimedia
    Commons submission."""

    tree = ElementTree.fromstring(wikixml)

    info = {}
    
    # Wikimedia Commons stores their information in a transcluded template
    # called "Information".
    for tmpl in tree.iter("template"):
        if tmpl.find("title").text:
            title = tmpl.find("title").text.strip()
            #Wikimedia Commons has separate info and artwork templates, but they
            #both are structured similarly enough to use.
            if title != "Information" and title != "Artwork":
                continue
        else:
            continue

        for part in tmpl.findall("part"):
            name = part.find("name").text.strip()
            value = part.find("value")

            # First, check if our value has language templates.
            # If so, pick out one.
            langs = extract_languages_from_wikinode(value)
            lang_value = None
            if len(langs) > 0:
                if preferred_lang in langs:
                    lang_value = langs[preferred_lang]
                else:
                    for lang in langs:
                        lang_value = langs[lang]
                        break
            else:
                lang_value = [value]
            
            #Now grab the actual data within.
            info[name] = extract_text_from_language_value(lang_value)
        
        break
    else:
        #raise Exception("Could not find information block in wikitext")
        pass

    return info