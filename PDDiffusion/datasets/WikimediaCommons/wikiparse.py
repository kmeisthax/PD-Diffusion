from defusedxml import ElementTree

def extract_languages_from_wikinode(wikinode, warn=False):
    """Given a node in a parsed wikitext XML file, find all language template
    invocations in the value and return them.
    
    This does not consider nested template invocations or templates buried in
    other styling.
    
    Wikimedia Commons has several forms of multilingual data:
    
     * ISO language code templates for individual language/value pairs
     * LangSwitch, which stores multiple languages and values (and, in
       MediaWiki, renders a LANGuage SWITCHer).
     * title and other templates which are intended to render values, but are
       also multilingual."""

    langs = {}

    for tmpl in wikinode.findall("template"):
        lang = tmpl.find("title").text.strip()
        
        if lang == "title":
            slot1 = None
            slot1lang = None

            for part in tmpl.findall("part"):
                inner_lang = part.find("name").text.strip()
                value = part.find("value")

                if inner_lang == "1":
                    slot1 = value
                elif inner_lang == "lang":
                    slot1lang = value.text
                elif inner_lang == "translation":
                    if warn:
                        print("title translation part values not yet supported")
                elif inner_lang == "transliteration":
                    if warn:
                        print("title transliteration part values not yet supported")
                else:
                    langs[inner_lang] = value
            
            if slot1lang is not None and slot1 is not None:
                langs[slot1lang] = slot1
            
            continue
        
        if lang == "LangSwitch":
            for part in tmpl.findall("part"):
                inner_lang = part.find("name").text.strip()
                value = part.find("value")

                if len(inner_lang) == 2:
                    langs[inner_lang] = value
                else:
                    if warn:
                        print(f"LangSwitch parameter {inner_lang} not yet supported")
            
            continue

        #We've accounted for all the 'special' templates, the rest is just
        #langcode templates.
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

                #Wikimedia Commons has a large number of templates that exist
                #purely to translate terms.
                #TODO: Actually extract the translations from the templates in
                #question instead of using their English titles
                if subtmpl_title.lower().startswith("creator:"):
                    true_value = (true_value + " " + subtmpl_title.removeprefix("Creator:").removeprefix("creator:")).strip()

                    if len(subtmpl.findall("part")) > 0:
                        if warn:
                            print(f"Creator template {true_value} has unknown data")
                elif subtmpl_title.lower().startswith("institution:"):
                    true_value = (true_value + " " + subtmpl_title.removeprefix("Institution:").removeprefix("institution:")).strip()

                    if len(subtmpl.findall("part")) > 0:
                        if warn:
                            print(f"Institution template {true_value} has unknown data")
                elif subtmpl_title.lower() == "Oil on canvas":
                    true_value = "Oil on canvas"
                elif subtmpl_title.lower() == "Portrait of male":
                    true_value = "Portrait of male"
                elif subtmpl_title.lower() == "Portrait of female":
                    true_value = "Portrait of female"
                elif subtmpl_title.lower() == "Portrait of a woman":
                    true_value = "Portrait of a woman"
                elif subtmpl_title.lower() == "Madonna and Child":
                    true_value = "Madonna and Child"
                else:
                    if warn:
                        print(f"Unknown value template {subtmpl_title}")
            else:
                if warn:
                    print(f"Unknown value tag {subtmpl.tag}")
            
            if subtmpl.tail:
                true_value = (true_value + " " + subtmpl.tail).strip()
    
    return true_value.strip()


def extract_information_from_wikitext(wikixml, warn=False, preferred_lang = "en"):
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
            langs = extract_languages_from_wikinode(value, warn=warn)
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
            info[name.lower()] = extract_text_from_language_value(lang_value, warn=warn)
        
        break
    else:
        #raise Exception("Could not find information block in wikitext")
        pass

    return info