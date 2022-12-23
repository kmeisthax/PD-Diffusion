"""Data augmentation strategies for AI training."""

import random, math

INTERESTING_KEYS=[
    "object type", "medium", "artist", "author", "date", "year", "title",
    "genre", "description", "subject", "subjects", "depicted people", "place",
    "depicted place", "depicted locations", "inscriptions", "technique",
    "keywords", "note", "extra info", "__label", "__pagetitle"
]

def data_row_is_valid(data_item, i, key):
    """Determine if a data row's key is valid for this particular item index or not."""
    if key not in INTERESTING_KEYS:
        return False
    
    if data_item[key] == "": #single item case
        return False
    
    if type(data_item) is not str and type(data_item) is not int and data_item[key][i] == "":
        return False
    
    return True

def augment_labels(data_item, min_range=0, max_range=None, num_augments=1):
    """Construct one or more randomly generated labels from the data in the data item.
    
    This assumes Wikimedia Commons format data; i.e. dict keys like:
    
     - object type, medium
     - artist, author
     - date, year
     - title
     - genre
     - description
     - subjects, subject, depicted people
     - place, depicted place, depicted locations
     - inscriptions
     - technique
     - keywords, note
     - extra info
     - __label
    
    Each item may either be a string or a list. Returns a list of strings, each
    one corresponding to a shuffling of the strings in that index in each key.
    
    min_range and max_range allow you to control batching of the labels. For
    example, if you have an entire dataset, but only want items 16 thru 32, set
    those two parameters to 16 and 32, respectively. If unset, we will process
    all the data in the data item.
    
    num_augments controls how many augments to return. If greater than one,
    then multiple labels will be created continuously and output one after
    another. Visually, that's like going from this:
    
        ["a", "b", "c", "d"]
        
    to:
    
        ["a", "a", "b", "b", "c", "c", "d", "d"]
    
    """
    
    #Is there anything useful in other_fields_1, other_fields_2, other_versions?
    #type? extra info?

    if max_range == None:
        max_range = 0
        for key in INTERESTING_KEYS:
            if type(data_item[key]) == str or type(data_item[key]) == int:
                max_range = max(max_range, 1)
            else:
                max_range = max(max_range, len(data_item[key]))
    
    output = []
    
    for i in range(min_range, max_range):
        keys = list(filter(lambda k: data_row_is_valid(data_item, i, k), list(data_item.keys())))

        for j in range(0, num_augments):
            random.shuffle(keys)
            out_value = []
            
            for key in keys[0:math.ceil(random.random() * len(keys))]:
                if type(data_item[key]) == str or type(data_item[key]) == int:
                    out_value.append(data_item[key])
                else:
                    out_value.append(data_item[key][i])
            
            output.append(", ".join(out_value))
    
    return output

def all_labels_in_item(data_item):
    """Return all the labels in a data item."""
    
    keys = list(filter(lambda k: k in INTERESTING_KEYS and data_item[k] != "", list(data_item.keys())))
    max_range = 0
    for key in keys:
        if type(data_item[key]) == str or type(data_item[key]) == int:
            max_range = max(max_range, 1)
        else:
            max_range = max(max_range, len(data_item[key]))
    
    labels = []
    for i in range(0, max_range):
        out_value = []

        for key in keys:
            if type(data_item[key]) == str or type(data_item[key]) == int:
                out_value.append(data_item[key])
            else:
                out_value.append(data_item[key][i])
        
        labels.append(", ".join(out_value))

    return labels