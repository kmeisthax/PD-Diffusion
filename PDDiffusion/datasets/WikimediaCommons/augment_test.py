from PDDiffusion.datasets.WikimediaCommons import local_wikimedia_base
from PDDiffusion.datasets.augment import augment_labels, all_labels_in_item

for item in local_wikimedia_base(load_images=False):
    print(augment_labels(item))
    print(augment_labels(item))
    print(augment_labels(item))
    print(all_labels_in_item(item))
    print("...")