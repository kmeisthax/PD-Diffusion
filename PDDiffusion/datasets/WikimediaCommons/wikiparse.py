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
    elif lang == "LangSwitch":
        for part in tmpl.findall("part"):
            inner_lang = part.find("name").text.strip()
            value = part.find("value")

            if len(inner_lang) == 2:
                langs[inner_lang] = value
            else:
                if warn:
                    print(f"LangSwitch parameter {inner_lang} not yet supported")
    elif len(lang) == 2:
        #Single language template, of the form {{lang|1=(language text)}}
        for part in tmpl.findall("part"):
            name = part.find("name")
            if "index" not in name or name.index != "1":
                continue
            
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
        for part in subtmpl.find("part"):
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