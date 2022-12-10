from PDDiffusion.datasets.WikimediaCommons.wikiparse import extract_information_from_wikitext
from PDDiffusion.datasets.WikimediaCommons import LOCAL_STORAGE
import itertools, glob, os, json, sys

report = []

for file in itertools.chain(
        glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpg")),
        glob.iglob(os.path.join(LOCAL_STORAGE, "*.jpeg")),
        glob.iglob(os.path.join(LOCAL_STORAGE, "*.png")),
        glob.iglob(os.path.join(LOCAL_STORAGE, "*.tif")),
        glob.iglob(os.path.join(LOCAL_STORAGE, "*.tiff"))
    ):
    
    if os.path.exists(file + ".json"):
        #Get around the fact that input redirection appears to change text encoding.
        print(file.encode("utf-8"))

        with open(file + ".json", 'r') as metadata_file:
            metadata = json.load(metadata_file)

            parses = {}

            for name in metadata["parsetree"]:
                xmlstr = metadata["parsetree"][name]

                try:
                    extracted = extract_information_from_wikitext(xmlstr, warn=True)

                    parses[name] = {
                        "status": "valid",
                        "data": extracted
                    }
                except Exception as e:
                    parses[name] = {
                        "status": "exception",
                        "message": str(e)
                    }
            
            report.append({
                "file": file,
                "status": "present",
                "parsetree": parses
            })
    else:
        report.append({
            "file": file,
            "status": "missing"
        })

with open("extract_test_results.json", 'w') as json_file:
    json.dump(report, json_file, indent=4)