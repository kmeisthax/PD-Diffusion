from defusedxml import ElementTree
import datetime

def print_warn(warning):
    print(warning.encode())

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

    lang = tmpl.find("title").text.strip().lower()
    
    if lang == "title" or lang == "alternative titles":
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
                        print_warn(f"Unknown title template parameter at index {name.attrib['index']}")
                    
                    continue
            
            if name.text is None:
                if warn:
                    print_warn("Skipping malformatted part of title template")
                
                continue

            inner_lang = name.text.strip()
            if inner_lang == "1":
                slot1 = value
            elif inner_lang == "lang":
                slot1lang = value.text
            elif inner_lang == "translation":
                if warn:
                    print_warn("title translation part values not yet supported")
            elif inner_lang == "transliteration":
                if warn:
                    print_warn("title transliteration part values not yet supported")
            else:
                langs[inner_lang] = value
        
        if slot1lang is not None and slot1 is not None:
            langs[slot1lang] = slot1
    elif lang == "langswitch":
        for part in tmpl.findall("part"):
            if part.find("name").text is None:
                if warn:
                    print_warn("Empty langswitch name, skipping")
                
                continue

            inner_lang = part.find("name").text.strip()
            value = part.find("value")

            if len(inner_lang) == 2:
                langs[inner_lang] = value
            else:
                if warn:
                    print_warn(f"LangSwitch parameter {inner_lang} not yet supported")
    elif lang == "inscription":
        #TODO: Some of the art history information is omitted here.
        params = extract_template_arguments(tmpl)
        
        slot1 = None
        slot1lang = None

        for key in params.keys():
            if len(key) == 2:
                langs[key] = params[key]
            elif key == "1":
                slot1 = params[key]
            elif key == "language" or key == "lang":
                slot1lang = params[key]
            else:
                if warn and (key != "type" or key != "position"): #ignore type/position since we can't translate it
                    print_warn(f"Unknown inscription key {key}")
        
        if slot1 is not None:
            if slot1lang is not None and len(slot1lang) == 2: #language code
                langs[slot1lang] = slot1
            else: #nonlinguistic, unknown, or bilingual
                langs["*"] = slot1
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

def parse_upper_and_lower_dates(lower_date, upper_date, warn=False):
    if lower_date.startswith("("):
        if warn:
            print_warn(f"first otherdate parameter {lower_date} is misformatted, please edit upstream")
        
        lower_date = lower_date.removeprefix("(").strip()
    
    if lower_date.startswith(":"):
        if warn:
            print_warn(f"first otherdate parameter {lower_date} is misformatted, please edit upstream")
        
        lower_date = lower_date.removeprefix(":").strip()
    
    if lower_date.endswith(";"):
        if warn:
            print_warn(f"first otherdate parameter {lower_date} is misformatted, please edit upstream")
        
        lower_date = lower_date.removesuffix(";").strip()
    
    #Some dates are in YYYY-YYYY format without being split into two template parameters.
    #or YYYY–YYYY format, or YYYY—YYYY format. YES THOSE ARE ALL DIFFERENT CHARACTERS
    #Someone also used YYYY\YYYY format
    lower_date_emless = lower_date.replace("–", "-").replace("—", "-").replace("\\", "-").replace("/", "-")
    if upper_date is None and '-' in lower_date_emless:
        #TODO: This introduces upper dates to formats that don't expect them.
        #They will be dropped for now
        lower_date_split = lower_date_emless.split("-")

        #Avoid trying to cut dates in YYYY-MM-DD or YYYY-MM format
        if len(lower_date_split) == 2:
            (maybe_lower_date, maybe_upper_date) = lower_date_split

            if int(maybe_upper_date) >= 13:
                if warn:
                    print_warn(f"Splitting double-year date {lower_date_emless} to {maybe_lower_date} and {maybe_upper_date}")
                
                lower_date = maybe_lower_date
                upper_date = maybe_upper_date
    
    #Sup dawg, I heard you like ambiguity
    if lower_date.startswith("c."):
        lower_date = lower_date.removeprefix("c.").strip()
    
    if upper_date is not None and upper_date.startswith("c."):
        upper_date = upper_date.removeprefix("c.").strip()
    
    return (lower_date, upper_date)

def parse_ymd(datestring):
    """Attempt to parse a date that may be missing a month or day number.
    
    Also returns a corresponding format string that prints out the provided
    fields as otherdate would print them in English.
    
    TODO: BC/AD"""
    try:
        return (datetime.datetime.strptime(datestring, "%Y-%m-%d"), "%d %B %Y")
    except ValueError:
        try:
            return (datetime.datetime.strptime(datestring, "%d %B %Y"), "%d %B %Y")
        except ValueError:
            try:
                return (datetime.datetime.strptime(datestring, "%Y-%m"), "%B %Y")
            except ValueError:
                try:
                    return (datetime.datetime.strptime(datestring, "%Y"), "%Y")
                except ValueError:
                    if "/" in datestring: #1735/36 style ambiguous dates
                        #TODO: We don't support date ranges, so just lop everything off
                        datestring = datestring.split("/")[0]
                    
                    if datestring.endswith("s"): #1700s style ambiguous dates
                        #TODO: This should be a date range, too
                        datestring = datestring[:-1]
                    
                    if " " in datestring: #1641 (1648?)
                        datestring = datestring.split(" ")[0]

                    #Datetime chokes on years before 1000 AD
                    return (datetime.datetime(int(datestring), 1, 1), "%Y")

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
        value = part.find("value")
        if value.find("template") is not None:
            if warn:
                print_warn("Nested otherdate templates are not supported")
            return ("", )
        
        if value.text is None: #Some templates have empty values
            if warn:
                print_warn("Otherdate template has empty value, skipping")
            continue
        
        value = value.text.strip()

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
            print_warn("Otherdate is missing notation type or lower date!")
        
        return ("",)

    if lower_date.endswith("century"): #{{otherdate|2quarter|century|15}}, {{otherdate|1sthalf|15th century}} and the like
        if warn:
            print_warn("Century otherdates are not supported")
        
        return ("",)
    
    if era is None:
        era = "AD"
    
    if warn and era == "BC":
        print_warn("otherdate BC mode is not fully supported")

    (lower_date, upper_date) = parse_upper_and_lower_dates(lower_date, upper_date, warn=warn)
    
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

            return (f"between circa {lower_date.strftime(lower_date_format)} and {upper_date.strftime(upper_date_format)}", lower_date, upper_date)
        else:
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

def mapreduce_languages(language_dict, operation):
    """Given a set of language values, do something to each language, which may
    or may not introduce more languages.
    
    The operation parameter will be called with the language value, plus the
    code for the value given. It may return either a string (mapping the value
    one-to-one per language) or another language dict. Nested language dicts
    are flattened to allow the operation to change the language provided,
    expose a child multilanguage dictionary, or indicate that it does not know
    the language of the text inside."""

    parsed_langs = {}

    for lang in language_dict.keys():
        this_lang_value = operation(language_dict[lang], lang)
        
        if type(this_lang_value) is dict:
            #Since templates are recursive they may also nest language
            #definitions. This is rare, but let's account for it!
            for thislang in this_lang_value.keys():
                if thislang == "*": #Unknown/any language
                    parsed_langs[lang] = this_lang_value["*"]
                else: #Language switched in template
                    parsed_langs[thislang] = this_lang_value[thislang]
        else:
            parsed_langs[lang] = this_lang_value
    
    return parsed_langs

def extract_template_arguments(subtmpl):
    """Get all the arguments out of a template and put them in a nice dict."""
    params = {}
    
    for part in subtmpl.findall("part"):
        name = part.find("name")
        value = part.find("value")

        name_text = name.text
        if (name.attrib is not None and "index" in name.attrib):
            name_text = name.attrib["index"]
        elif name_text is None:
            name_text = ""
        else:
            name_text = name_text.strip()
        
        params[name_text] = value
    
    return params

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
        #Language content may either be an unknown language (which we pass
        #through) or it may change inside of other templates.
        langs = mapreduce_languages(langs, lambda value, _: extract_text_from_value(value, warn=warn, preferred_lang=preferred_lang))

        if preferred_lang is not None:
            if preferred_lang in langs:
                return langs[preferred_lang]
            elif "*" in langs:
                return langs["*"]
            else:
                if warn:
                    print_warn(f"Language template is missing preferred language {preferred_lang}")
                
                return ""
        else:
            return langs
    
    #Wikimedia Commons has a large number of templates that exist purely to
    #translate terms. We call these "non-language templates".
    #TODO: Actually extract the translations from the templates in
    #question instead of using their English titles
    text_value = ""

    params = extract_template_arguments(subtmpl)
    
    if subtmpl_title.lower() == "information field":
        pass #Infofields are handled already in the toplevel fn
    elif subtmpl_title.lower().startswith("creator:"):
        text_value = (text_value + " " + subtmpl_title.removeprefix("Creator:").removeprefix("creator:")).strip()

        if len(subtmpl.findall("part")) > 0:
            if warn:
                print_warn(f"Creator template {text_value} has unknown data")
    elif subtmpl_title.lower().startswith("institution:"):
        text_value = (text_value + " " + subtmpl_title.removeprefix("Institution:").removeprefix("institution:")).strip()

        if len(subtmpl.findall("part")) > 0:
            if warn:
                print_warn(f"Institution template {text_value} has unknown data")
    elif subtmpl_title.lower() == "oil on canvas":
        text_value = "Oil on canvas"
    elif subtmpl_title.lower() == "oil on panel" or subtmpl_title.lower() == "oil on wood":
        text_value = "Oil on panel"
    elif subtmpl_title.lower() == "tempera on panel" or subtmpl_title.lower() == "tempera on wood" or subtmpl_title.lower() == "tempera on wood panel":
        text_value = "Tempera on panel"
    elif subtmpl_title.lower() == "fresco":
        text_value = "Fresco"
    elif subtmpl_title.lower() == "oil on copper" or subtmpl_title.lower() == "oil on metal" or subtmpl_title.lower() == "oil on metal plate":
        text_value = "Oil on copper"
    elif subtmpl_title.lower() == "portrait of male" or subtmpl_title.lower() == "portrait of female" or subtmpl_title.lower() == "portrait of":
        #Portrait-of templates can contain multilingual contents.
        for name_text in params.keys():
            if name_text == "1":
                text_value = extract_text_from_value(params[name_text], warn=warn, preferred_lang="en")
            else:
                if warn:
                    print_warn(f"Unknown portrait of template parameter {name_text}")

        text_value = f"Portrait of {text_value}"
    elif subtmpl_title.lower() == "technique":
        text_value = ""

        for name_text in params.keys():
            value = params[name_text]

            if name_text == "1":
                text_value += extract_text_from_value(value, warn=warn, preferred_lang="en")
            elif name_text == "and":
                text_value += " and " + extract_text_from_value(value, warn=warn, preferred_lang="en")
            elif name_text == "2" or name_text == "on":
                text_value += " on " + extract_text_from_value(value, warn=warn, preferred_lang="en")
            else:
                if warn:
                    print_warn(f"Unsupported technique template parameter {name_text}")
    elif subtmpl_title.lower() == "portrait of a woman":
        text_value = "Portrait of a woman"
    elif subtmpl_title.lower() == "madonna and child":
        text_value = "Madonna and Child"
    elif subtmpl_title.lower() == "assumption of mary":
        text_value = "Assumption of Mary"
    elif subtmpl_title.lower() == "and" or subtmpl_title.lower() == "Conj-and":
        text_value = "and"
    elif subtmpl_title.lower() == "drawing":
        text_value = "drawing"
    elif subtmpl_title.lower() == "sheet":
        text_value = "Sheet"
    elif subtmpl_title.lower() == "engraving":
        text_value = "engraving"
    elif subtmpl_title.lower() == "anonymous":
        text_value = "Anonymous"
    elif subtmpl_title.lower() == "unknown":
        text_value = "Unknown"
        
        for name_text in params.keys():
            value = params[name_text]

            if value is not None and value.text is not None:
                text_value = f"Unknown {value.text.strip()}"
    elif subtmpl_title.lower() == "own":
        text_value = "Own work"
    elif subtmpl_title.lower() == "own photo" or subtmpl_title.lower() == "self-photographed":
        text_value = "Self-photographed"
    elif subtmpl_title.lower() == "own work by original uploader":
        text_value = "Own work by the original uploader"
    elif subtmpl_title.lower() == "between":
        lower_date = extract_text_from_value(params["1"], warn=warn, preferred_lang="en")
        upper_date = None

        if "2" in params:
            upper_date = extract_text_from_value(params["2"], warn=warn, preferred_lang="en")
        else:
            if warn:
                print_warn(f"Missing upper date for between template, lower date is {lower_date}. Fix upstream!")

        (lower_date, upper_date) = parse_upper_and_lower_dates(lower_date, upper_date, warn=warn)

        lower_date = parse_ymd(lower_date)[0]
        upper_date = parse_ymd(upper_date)[0]
        
        text_value = f"from {lower_date} to {upper_date}"
    elif subtmpl_title.lower() == "circa": #TODO BC/AD
        if "1" in params:
            (lower_year, upper_year) = parse_upper_and_lower_dates(extract_text_from_value(params["1"], warn=warn, preferred_lang="en"), None, warn=warn)
            try:
                y = int(lower_year)

                if "2" in params:
                    m = int(extract_text_from_value(params["2"], warn=warn, preferred_lang="en"))
                    if m > 12:
                        text_value = f"circa {y}-{m}"
                    elif "3" in params:
                        d = int(extract_text_from_value(params["3"], warn=warn, preferred_lang="en"))

                        text_value = "circa " + datetime.datetime(y, m, d).strftime("%d %B %Y")
                    else:
                        text_value = "circa " + datetime.datetime(y, m, 1).strftime("%B %Y")
                elif upper_year is not None:
                    y2 = int(upper_year)
                    text_value = f"circa {y}-{y2}"
                else:
                    text_value = f"circa {y}"
            except ValueError: #Some users of circa don't actually put a valid date into the year slot
                value = extract_text_from_value(params["1"], warn=warn, preferred_lang="en")
                text_value = f"circa {value}"
        else:
            if warn:
                print_warn("Missing circa date. Fix upstream!")
            text_value = "circa"
    elif subtmpl_title.lower() == "other date" or subtmpl_title.lower() == "otherdate":
        text_value = evaluate_otherdate(subtmpl, warn=warn)[0]
    elif subtmpl_title.lower() == "ucfirst:" or subtmpl_title.lower() == "ucfirstletter:":
        #ucfirst actually puts its value in the title, somehow
        text_value = extract_template_tag(subtmpl.find("title").find("template"), warn=warn)
    elif subtmpl_title.lower() == "size":
        unit = None
        if "unit" in params:
            unit = extract_text_from_value(params["unit"], warn=warn, preferred_lang="en")
        
        if "1" in params:
            unit = extract_text_from_value(params["1"], warn=warn, preferred_lang="en")
        
        if unit is not None:
            dimensions = []
            #Deprecated dimensional format
            if "2" in params:
                dimensions.append(extract_text_from_value(params['2'], warn=warn, preferred_lang='en'))

                if "3" in params:
                    dimensions.append(extract_text_from_value(params['3'], warn=warn, preferred_lang='en'))
                
                if "4" in params:
                    dimensions.append(extract_text_from_value(params['4'], warn=warn, preferred_lang='en'))
                
                text_value = f"{' × '.join(dimensions)} {unit}"
            else:
                for name in params.keys():
                    if name == "unit":
                        continue
                    
                    value = params[name]
                    value = extract_text_from_value(value, warn=warn, preferred_lang='en')
                    
                    dimensions.append(f"{name}: {value} {unit}")
                
                text_value = "; ".join(dimensions)
        else:
            if warn:
                print_warn("Size template is missing its unit parameter! Fix upstream!")
    elif subtmpl_title.lower() == "geographicus-link":
        value = extract_text_from_value(params["1"], warn=warn, preferred_lang='en')
        text_value = f"Geographicus link {value}"
    elif subtmpl_title.lower() == "geographicus-source":
        text_value = "This file was provided to Wikimedia Commons by Geographicus Rare Antique Maps, a specialist dealer in rare maps and other cartography of the 15th, 16th, 17th, 18th and 19th centuries, as part of a cooperation project."
    elif subtmpl_title.lower() == "sourcenpglondon":
        #TODO: This indicates an ACTIVE LEGAL THREAT to wikimedia commons based
        #off the UK's sweat-of-the-brow crap.
        text_value = "While Commons policy accepts the use of this media, one or more third parties have made copyright claims against Wikimedia Commons in relation to the work from which this is sourced or a purely mechanical reproduction thereof."
    elif subtmpl_title.lower() == "pd-art" or subtmpl_title.lower() == "pd-art-old-100":
        #Note: this is not the whole template but we're going to be parsing license data in other ways
        text_value = "This is a faithful photographic reproduction of a two-dimensional, public domain work of art."
    elif subtmpl_title.lower() == "self":
        #TODO: This also has license assertions that may counteract PD-Art.
        text_value = "I, the copyright holder of this work, hereby publish it under the following license:"
    elif subtmpl_title.lower() == "loc-map":
        text_value = "This map is available from the United States Library of Congress's Geography & Map Division"
    elif subtmpl_title.lower() == "royal museums greenwich":
        #TODO: Implies CC-BY-NC-SA on the labels
        text_value = "The original artefact or artwork has been assessed as public domain by age, and faithful reproductions of the two dimensional work are also public domain. No permission is required for reuse for any purpose. The text of this image record has been derived from the Royal Museums Greenwich catalogue and image metadata. Individual data and facts such as date, author and title are not copyrightable, but reuse of longer descriptive text from the catalogue may not be considered fair use. Reuse of the text must be attributed to the 'National Maritime Museum, Greenwich, London' and a Creative Commons CC-BY-NC-SA-3.0 license may apply if not rewritten. Refer to Royal Museums Greenwich copyright."
    elif subtmpl_title.lower() == "rkd":
        text_value = "This image is available from the Netherlands Institute for Art History."
    elif subtmpl_title.lower() == "not on view":
        text_value = "not on view"
    elif subtmpl_title.lower() == "private collection": #TODO: There's extra arguments here.
        text_value = "Private collection"
    elif subtmpl_title.lower() == "extracted from" or subtmpl_title.lower() == "ef" or subtmpl_title.lower() == "cropped":
        value = extract_text_from_value(params["1"], warn=warn, preferred_lang='en')
        text_value = f"This file has been extracted from another file: {value}"
    elif subtmpl_title.lower() == "image extracted" or subtmpl_title.lower() == "extracted":
        value = extract_text_from_value(params["1"], warn=warn, preferred_lang='en')
        text_value = f"This file has an extracted image: {value}"
    elif subtmpl_title.lower() == "detail":
        if "1" in params:
            description = extract_text_from_value(params["1"], warn=warn, preferred_lang='en')

            if "position" in params:
                position = extract_text_from_value(params["position"], warn=warn, preferred_lang='en')
                text_value = f"detail: {description} ({position})"
            else:
                text_value = f"detail: {description}"
        else:
            if warn:
                print_warn("Detail template is missing description")
    elif subtmpl_title.lower() == "wga link":
        pic_url = ""
        if "pic-url" in params:
            pic_url = extract_text_from_value(params["pic-url"], warn=warn, preferred_lang='en')
        
        info_url = ""
        if "info-url" in params:
            info_url = extract_text_from_value(params["info-url"], warn=warn, preferred_lang='en')
        
        text_value = f"Web Gallery of Art: {pic_url} {info_url}"
    elif subtmpl_title.lower() == "w":
        text_value = "Main Page"
        if "1" in params:
            text_value = extract_text_from_value(params["1"], warn=warn, preferred_lang='en')
        
        if "2" in params:
            text_value = extract_text_from_value(params["2"], warn=warn, preferred_lang='en')
    elif subtmpl_title.lower() == "provenanceevent":
        #TODO: Actually parse the provenance event template
        text_value = ""
    else:
        if warn:
            print_warn(f"Unknown value template {subtmpl_title}")
    
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
                print_warn(f"Unknown value tag {lang_value.tag}")
        
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

def extract_other_fields_from_value(value, warn=False, preferred_lang="en"):
    """Extract key-value pairs from the "other fields" part of an info/artwork template."""
    extra_params = {}

    for tmpl in value.findall("template"):
        title = tmpl.find("title").text.strip()
        if title.lower() != "information field":
            continue

        params = extract_template_arguments(tmpl)
        
        name = None
        if "name" in params:
            name = params["name"]
        elif "Name" in params:
            name = params["Name"]
        elif "1" in params:
            name = params["1"]
        
        value = None
        if "value" in params:
            value = params["value"]
        elif "Value" in params:
            value = params["Value"]
        elif "2" in params:
            value = params["2"]

        name = extract_text_from_value(name, warn=warn, preferred_lang="en") #Name is a key so we need a string.
        if value is not None:
            value = extract_text_from_value(value, warn=warn)
        else:
            if warn:
                print_warn(f"Information field {name} is missing its value")
            
            value = ""
        
        extra_params[name] = [value]
    
    return extra_params

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
            name = part.find("name")
            value = part.find("value")
            
            if name.text is None:
                if warn:
                    print_warn("Found a spurious top-level template argument")
                continue

            name = name.text.strip()

            #Info fields can live anywhere and jump out into the info block
            extra_fields = extract_other_fields_from_value(value, warn=warn, preferred_lang=preferred_lang)
            for name in extra_fields.keys():
                info[name] = extra_fields[name]
            
            info[name.lower()] = extract_text_from_value(value, warn=warn, preferred_lang=preferred_lang)
        
        break
    else:
        #raise Exception("Could not find information block in wikitext")
        pass

    return info