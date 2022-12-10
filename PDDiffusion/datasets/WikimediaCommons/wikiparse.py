from defusedxml import ElementTree
import datetime

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

def parse_ymd(datestring):
    """Attempt to parse a date that may be missing a month or day number.
    
    Also returns a corresponding format string that prints out the provided
    fields as otherdate would print them in English.
    
    TODO: BC/AD"""
    try:
        return (datetime.datetime.strptime(datestring, "%Y-%m-%d"), "%d %B %Y")
    except ValueError:
        try:
            return datetime.datetime.strptime(datestring, "%Y-%m", "%B %Y")
        except ValueError:
            return datetime.datetime.strptime(datestring, "%Y", "%Y")

def evaluate_otherdate(wikinode, warn=False):
    """Emulate the 'otherdate' template.

    `wikinode` should be the otherdate template.
    
    Returns a list of up to three values: the text string itself, then era
    (BC/AD), plus up to two date strings in the value.
    
    Note: this only works for English language values.
    Also note: This only returns Gregorian dates, even if the original template
    value had other calendar dates in it."""

    notation_type = None
    lower_date = None
    upper_date = None
    era = None

    for part in wikinode.findall("part"):
        value = part.find("value").text.strip()

        if "index" in part.find("name"):
            index = part.find("name").index

            if index == "1":
                notation_type = value
            elif index == "2":
                lower_date = value
            elif index == "3":
                upper_date = value
        else:
            argname = part.find("name").text.strip()

            if argname == "era":
                era = value
    
    if notation_type is None or lower_date is None:
        if warn:
            print("Otherdate is missing notation type or lower date!")
        
        return ("")
    
    if era is None:
        era = "AD"
    
    if notation_type.lower() == "islamic":
        #Islamic dates store the Gregorian equivalent in the lower slot and the
        #Islamic calendar original in the upper

        if upper_date is not None:
            #TODO: convert BC/AD to BH/AH
            #NOTE: We do not yield the Islamic equivalent, just the Gregorian date
            return (f"{lower_date} ({upper_date})", era, parse_ymd(lower_date)[0])
        else:
            return (f"{lower_date}", era, parse_ymd(lower_date)[0])
    elif notation_type.lower() == "-" or notation_type.lower() == "from-until":
        (lower_date, lower_date_format) = parse_ymd(lower_date)

        if upper_date is not None:
            (upper_date, upper_date_format) = parse_ymd(upper_date)

            return (f"from {lower_date.strftime(lower_date_format)} until {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        else:
            return (f"from {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "~" or notation_type.lower() == "ca" or notation_type.lower() == "circa":
        (lower_date, lower_date_format) = parse_ymd(lower_date)

        if upper_date is not None:
            (upper_date, upper_date_format) = parse_ymd(upper_date)

            return (f"from {lower_date.strftime(lower_date_format)} until {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        else:
            return (f"from {lower_date.strftime(lower_date_format)}", lower_date)
    else:
        if warn:
            print (f"Unsupported otherdate notation {notation_type}")
        
        return ("")


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
                elif subtmpl_title.lower() == "oil on canvas":
                    true_value = "Oil on canvas"
                elif subtmpl_title.lower() == "oil on panel":
                    true_value = "Oil on panel"
                elif subtmpl_title.lower() == "portrait of male":
                    true_value = "Portrait of male"
                elif subtmpl_title.lower() == "portrait of female":
                    true_value = "Portrait of female"
                elif subtmpl_title.lower() == "portrait of a woman":
                    true_value = "Portrait of a woman"
                elif subtmpl_title.lower() == "madonna and child":
                    true_value = "Madonna and Child"
                elif subtmpl_title.lower() == "unknown":
                    for part in subtmpl.find("part"):
                        true_value = f"Unknown {part.find('value').text.strip()}"
                elif subtmpl_title.lower() == "other date" or subtmpl_title.lower() == "otherdate":
                    true_value = evaluate_otherdate(subtmpl, warn=warn)[0]
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