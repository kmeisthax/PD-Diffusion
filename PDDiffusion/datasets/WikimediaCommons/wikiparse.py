from defusedxml import ElementTree
import datetime

def extract_languages_from_template(tmpl, warn=False):
    """Given a template element in a parsed wikitext XML file, extract all
    translations in the template's arguments.

    If the template does not contain translation arguments (either because the
    term is not translatable or because the template is supposed to translate
    things internally), this returns None.
    
    Dict values will be XML nodes - specifically, <value> tags potentially
    containing other templates inside. Nested template invocations will need to
    be considered separately.
    
    Wikimedia Commons has several forms of multilingual data:
    
     * ISO language code templates for individual language/value pairs
     * LangSwitch, which stores multiple languages and values (and, in
       MediaWiki, renders a LANGuage SWITCHer).
     * title and other templates which are intended to render values, but are
       also multilingual."""

    langs = {}

    lang = tmpl.find("title").text.strip()
    
    if lang == "title":
        slot1 = None
        slot1lang = None

        for part in tmpl.findall("part"):
            name = part.find("name")
            value = part.find("value")

            #Slot 1 can either be by index or by keyword
            if name.text is None and "index" in name.attrib:
                if name.attrib["index"] == "1":
                    slot1 = value
                    continue
                else:
                    if warn:
                        print(f"Unknown title template parameter at index {name.attrib['index']}")
                    
                    continue

            inner_lang = name.text.strip()
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
    elif lang == "LangSwitch":
        for part in tmpl.findall("part"):
            if part.find("name").text is None:
                if warn:
                    print("Empty langswitch name, skipping")
                
                continue

            inner_lang = part.find("name").text.strip()
            value = part.find("value")

            if len(inner_lang) == 2:
                langs[inner_lang] = value
            else:
                if warn:
                    print(f"LangSwitch parameter {inner_lang} not yet supported")
    elif len(lang) == 2:
        #Single language template, of the form {{lang|1=(language text)}}
        #or {{lang|(language text)}}
        for part in tmpl.findall("part"):
            name = part.find("name")
            if name.text is not None and name.text.strip() == "1":
                langs[lang] = part.find("value")
                break
            elif "index" in name.attrib and name.attrib["index"] == "1":
                langs[lang] = part.find("value")
                break
    else: #Non-language template
        return None
    
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
            return (datetime.datetime.strptime(datestring, "%Y-%m"), "%B %Y")
        except ValueError:
            return (datetime.datetime.strptime(datestring, "%Y"), "%Y")

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

        if "index" in part.find("name").attrib:
            index = part.find("name").attrib["index"]

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
        
        return ("",)
    
    if era is None:
        era = "AD"
    
    if warn and era == "BC":
        print("otherdate BC mode is not fully supported")
    
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
        if upper_date is not None:
            (lower_date, lower_date_format) = parse_ymd(lower_date)
            (upper_date, upper_date_format) = parse_ymd(upper_date)

            return (f"between circa {lower_date.strftime(lower_date_format)} and {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        elif "-" in lower_date and len(lower_date) == 9:
            #Some date pairs are in the form of YYYY-YYYY
            (lower_date, upper_date) = lower_date.split("-")

            (lower_date, lower_date_format) = parse_ymd(lower_date)
            (upper_date, upper_date_format) = parse_ymd(upper_date)

            return (f"between circa {lower_date.strftime(lower_date_format)} and {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        else:
            (lower_date, lower_date_format) = parse_ymd(lower_date)

            return (f"circa {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "between":
        (lower_date, lower_date_format) = parse_ymd(lower_date)

        if upper_date is not None:
            (upper_date, upper_date_format) = parse_ymd(upper_date)

            return (f"between {lower_date.strftime(lower_date_format)} and {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        else:
            return (f"between {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "or":
        (lower_date, lower_date_format) = parse_ymd(lower_date)

        if upper_date is not None:
            (upper_date, upper_date_format) = parse_ymd(upper_date)

            return (f"{lower_date.strftime(lower_date_format)} or {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        else:
            return (f"{lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "and" or notation_type.lower() == "&":
        (lower_date, lower_date_format) = parse_ymd(lower_date)

        if upper_date is not None:
            (upper_date, upper_date_format) = parse_ymd(upper_date)

            return (f"{lower_date.strftime(lower_date_format)} and {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        else:
            return (f"{lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "by":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"by {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "from":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"from {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "until":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"until {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "before" or notation_type.lower() == "b" or notation_type.lower() == "<":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"before {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "after" or notation_type.lower() == "a" or notation_type.lower() == ">":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"after {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "spring":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"Spring {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "summer":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"Summer {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "fall" or notation_type.lower() == "autumn":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"Autumn {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "winter":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"Winter {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "beginning" or notation_type.lower() == "early":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"early {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "mid" or notation_type.lower() == "middle":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"mid {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "end" or notation_type.lower() == "late":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"late {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "firsthalf" or notation_type.lower() == "1half" or notation_type.lower() == "1sthalf" or notation_type.lower() == "first half" or notation_type.lower() == "1st half":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"first half of {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "secondhalf" or notation_type.lower() == "2half" or notation_type.lower() == "2ndhalf" or notation_type.lower() == "second half" or notation_type.lower() == "2nd half":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"second half of {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "1quarter" or notation_type.lower() == "quarter1" or notation_type.lower() == "1stquarter" or notation_type.lower() == "1st quarter":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"first quarter of {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "2quarter" or notation_type.lower() == "quarter2" or notation_type.lower() == "2ndquarter" or notation_type.lower() == "2nd quarter":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"second quarter of {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "3quarter" or notation_type.lower() == "quarter3" or notation_type.lower() == "3rdquarter" or notation_type.lower() == "3rd quarter":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"third quarter of {lower_date.strftime(lower_date_format)}", lower_date)
    elif notation_type.lower() == "4quarter" or notation_type.lower() == "quarter4" or notation_type.lower() == "4thquarter" or notation_type.lower() == "4th quarter":
        (lower_date, lower_date_format) = parse_ymd(lower_date)
        return (f"fourth quarter of {lower_date.strftime(lower_date_format)}", lower_date)
    else:
        if warn:
            print (f"Unsupported otherdate notation {notation_type}")
        
        return ("",)

def extract_template_tag(subtmpl, warn=False, preferred_lang="en"):
    """Given a template tag, extract its value.
    
    If this template is a language tag, the preferred language value will be
    extracted from it, unless preferred_lang is None, in which case all lang
    values will be returned as a dict.
    
    If the preferred language is not available, this returns the empty string."""
    subtmpl_title = subtmpl.find("title").text.strip()

    langs = extract_languages_from_template(subtmpl, warn=warn)
    if langs is not None:
        #Each language itself is a template value which we need to reparse.
        if preferred_lang is None:
            parsed_langs = {}

            for lang in langs.keys():
                this_lang_value = extract_text_from_value(langs[lang], warn=warn, preferred_lang=preferred_lang)

                #Since templates are recursive they may also nest language
                #definitions. This is rare, but let's account for it!
                for thislang in this_lang_value.keys():
                    if thislang == "*": #Unknown/any language
                        parsed_langs[lang] = this_lang_value["*"]
                    else: #Language switched in template
                        parsed_langs[thislang] = this_lang_value[thislang]
            
            return parsed_langs
        elif preferred_lang in langs:
            #NOTE: This may fail in the case of badly nested language templates.
            #For example, {{ja |1= {{en |1=hello}}}} is a valid template
            #construction, but we won't visit the English tag here.
            return extract_text_from_value(langs[preferred_lang], warn=warn, preferred_lang=preferred_lang)
        else:
            if warn:
                print(f"Language template is missing preferred language {preferred_lang}")
            
            return ""
    
    #Wikimedia Commons has a large number of templates that exist purely to
    #translate terms. We call these "non-language templates".
    #TODO: Actually extract the translations from the templates in
    #question instead of using their English titles
    text_value = ""
    
    if subtmpl_title.lower().startswith("creator:"):
        text_value = (text_value + " " + subtmpl_title.removeprefix("Creator:").removeprefix("creator:")).strip()

        if len(subtmpl.findall("part")) > 0:
            if warn:
                print(f"Creator template {text_value} has unknown data")
    elif subtmpl_title.lower().startswith("institution:"):
        text_value = (text_value + " " + subtmpl_title.removeprefix("Institution:").removeprefix("institution:")).strip()

        if len(subtmpl.findall("part")) > 0:
            if warn:
                print(f"Institution template {text_value} has unknown data")
    elif subtmpl_title.lower() == "oil on canvas":
        text_value = "Oil on canvas"
    elif subtmpl_title.lower() == "oil on panel":
        text_value = "Oil on panel"
    elif subtmpl_title.lower() == "portrait of male":
        text_value = "Portrait of male"
    elif subtmpl_title.lower() == "portrait of female":
        text_value = "Portrait of female"
    elif subtmpl_title.lower() == "portrait of a woman":
        text_value = "Portrait of a woman"
    elif subtmpl_title.lower() == "madonna and child":
        text_value = "Madonna and Child"
    elif subtmpl_title.lower() == "unknown":
        text_value = "Unknown"

        for part in subtmpl.find("part"):
            value = part.find('value')

            if value is not None and value.text is not None:
                text_value = f"Unknown {part.find('value').text.strip()}"
    elif subtmpl_title.lower() == "other date" or subtmpl_title.lower() == "otherdate":
        text_value = evaluate_otherdate(subtmpl, warn=warn)[0]
    elif subtmpl_title.lower() == "ucfirst:" or subtmpl_title.lower() == "ucfirstletter:":
        #ucfirst actually puts its value in the title, somehow
        text_value = extract_template_tag(subtmpl.find("title").find("template"), warn=warn)
    else:
        if warn:
            print(f"Unknown value template {subtmpl_title}")
    
    #This return path is only taken for templates that do not have multiple
    #language values as arguments. They can be translated, but we do not
    #currently support translating them, so they will always be returned as
    #English language strings.
    if preferred_lang is None:
        return {"en": text_value}
    elif preferred_lang != "en":
        #Since we don't translate non-language templates ourselves, they will
        #be omitted from all non-English output
        return ""
    
    return text_value

def extract_text_from_value(wikinodes, warn=False, preferred_lang="en"):
    """Given a value tag, extract all relevant text in a given language.
    
    If preferred_lang is None, then all languages will be extracted as a dict.
    Otherwise, extracted text will be returned as a string.
    
    Template values that do not have language tags will be marked with the
    special language "*" which means any language."""

    #Format of value_accum: Strings for untranslated text, dicts for translated
    #text (in preferred_lang=None mode). Everything has implicit spaces
    #surrounding it.
    value_accum = []

    #First, extract EVERYTHING WE CAN into the accumulator
    if wikinodes.text:
        value_accum.append(wikinodes.text)
    
    for lang_value in wikinodes:
        if lang_value.tag == "template":
            value_accum.append(extract_template_tag(lang_value, warn=warn, preferred_lang=preferred_lang))
        else:
            if warn:
                print(f"Unknown value tag {lang_value.tag}")
        
        if lang_value.tail:
            value_accum.append(lang_value.tail)
    
    #Now, we need to produce a list of languages to extract.
    target_langs = set()
    if preferred_lang is None:
        for value in value_accum:
            if type(value) is dict:
                for lang in value.keys():
                    target_langs.add(lang)
    else:
        target_langs.add(preferred_lang)
    
    #Failsafe: If we didn't find any language tags inside, report the value as
    #unknown.
    if len(target_langs) == 0:
        true_value = ""

        for value in value_accum:
            #We don't bother with dicts since they're already empty
            if type(value) is str:
                true_value = true_value + " " + value
        
        if preferred_lang is None:
            return {"*": true_value.strip()}
        else:
            return true_value.strip()

    #Finally, extract each language value separately.
    lang_values = {}

    for lang in target_langs:
        true_value = ""

        for value in value_accum:
            if type(value) is dict and lang in value:
                true_value = true_value + " " + value[lang]
            elif type(value) is str:
                true_value = true_value + " " + value
        
        lang_values[lang] = true_value.strip()
    
    if preferred_lang is None:
        return lang_values
    elif preferred_lang in lang_values:
        return lang_values[preferred_lang]
    else:
        return ""

def extract_information_from_wikitext(wikixml, warn=False, preferred_lang = "en"):
    """Extract info from a Mediawiki XML parse tree related to a Wikimedia
    Commons submission.
    
    All strings will be in the preferred language (if multiple translations are
    available. You may instead set preferred_lang to None if you want all
    available translation strings."""

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

            info[name.lower()] = extract_text_from_value(value, warn=warn, preferred_lang=preferred_lang)
        
        break
    else:
        #raise Exception("Could not find information block in wikitext")
        pass

    return info