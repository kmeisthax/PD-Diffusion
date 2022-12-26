"""Test extracting data from stored wikitext."""

import os, json, csv
from PDDiffusion.datasets.WikimediaCommons.wikiparse import extract_information_from_wikitext
from PDDiffusion.datasets.WikimediaCommons.model import WikimediaCommonsImage
from PDDiffusion.datasets.model import File
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

report = []

load_dotenv()
engine = create_engine(os.getenv("DATABASE_CONNECTION"), future=True)

with Session(engine) as session:
    for (article, image) in WikimediaCommonsImage.select_all_image_articles(session):
        if image.file.storage_provider != File.LOCAL_FILE:
            print(f"Non-local file provider {image.file.storage_provider}")

        file = image.file.url
        metadata = article.wikidata
        
        #Get around the fact that input redirection appears to change text encoding.
        print(file.encode("utf-8"))

        parses = {}

        for name in metadata["parsetree"]:
            xmlstr = metadata["parsetree"][name]

            try:
                extracted = extract_information_from_wikitext(xmlstr, warn=True, preferred_lang=None)

                parses[name] = {
                    "status": "valid",
                    "data": extracted,
                    "en_only": extract_information_from_wikitext(xmlstr, warn=True, preferred_lang="en")
                }
            except Exception as e:
                parses[name] = {
                    "status": "exception",
                    "message": str(e)
                }
        
        terms = None
        if "terms" in metadata:
            terms = metadata["terms"]
        
        report.append({
            "file": file,
            "terms": terms,
            "status": "present",
            "parsetree": parses
        })

with open("extract_test_results.json", 'w') as json_file:
    json.dump(report, json_file, indent=4)

headers = ["file", "object type", "artist", "date", "title", "medium", "description", "__label"]
headermap = {key: id for (id, key) in enumerate(headers)}

rows = []

for item in report:
    if item["status"] == "present":
        item_data = {"file": item["file"]}

        if item["terms"] is not None:
            for key in item["terms"].keys():
                item_data["__" + key] = ", ".join(item["terms"][key])

        item_data.update(item["parsetree"]["*"]["en_only"])

        for key in item_data.keys():
            if key not in headers:
                key_id = len(headers)
                headers.append(key)
                headermap[key] = key_id
        
        item_row = [""] * len(headers)
        
        for key in headermap.keys():
            if key in item_data:
                item_row[headermap[key]] = item_data[key]
        
        rows.append(item_row)

with open("extracted_data.csv", 'w', newline='', encoding="utf-8") as data_file:
    writer = csv.writer(data_file)

    writer.writerow(headers)
    
    for item in rows:
        writer.writerow(item)
