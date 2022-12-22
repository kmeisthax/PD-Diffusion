"""Data augmentation strategies for AI training."""

import random, math

def augment_labels(data_item):
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
    one corresponding to a shuffling of the strings in that index in each key."""
    
    #Is there anything useful in other_fields_1, other_fields_2, other_versions?
    #type? extra info?

    keys = list(filter(lambda k: k != "image" and k != "image_file_path" and k != "__pagetitle" and k != "__pageid" and data_item[k] != "", list(data_item.keys())))

    max_range = 0

    random.shuffle(keys)
    for key in keys:
        if type(data_item[key]) == str or type(data_item[key]) == int:
            max_range = max(max_range, 1)
        else:
            max_range = max(max_range, len(data_item[key]))
    
    output = []
    for i in range(0, max_range):
        out_value = []
        
        for key in keys[0:math.ceil(random.random() * len(keys))]:
            if type(data_item[key]) == str or type(data_item[key]) == int:
                out_value.append(data_item[key])
            elif i in data_item[key]:
                out_value.append(data_item[key][i])
        
        if len(out_value) == 0:
            #Emergency case
            if type(data_item["__pagetitle"]) == str:
                out_value.append(data_item["__pagetitle"])
            elif i in data_item["__pagetitle"]:
                out_value.append(data_item["__pagetitle"][i])
        
        output.append(", ".join(out_value))
    
    return output

def all_labels_in_item(data_item):
    """Return all the labels in a data item."""
    
    keys = filter(lambda k: k != "image" and k != "image_file_path" and k != "__pagetitle" and k != "__pageid" and data_item[k] != "", list(data_item.keys()))
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
        
        if len(out_value) == 0:
            #Emergency case
            if type(data_item["__pagetitle"]) == str:
                out_value.append(data_item["__pagetitle"])
            elif i in data_item["__pagetitle"]:
                out_value.append(data_item["__pagetitle"][i])
        
        labels.append(", ".join(out_value))

    return labels