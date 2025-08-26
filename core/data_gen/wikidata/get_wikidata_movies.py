from dotenv import load_dotenv

load_dotenv()
from qwikidata.sparql import return_sparql_query_results
import requests
import time

def run_sparql(sparql_query):

    # Wikidata Query Service endpoint URL
    wikidata_endpoint = "https://query.wikidata.org/sparql"

    # Set up headers for the API request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }

    # Make the API request
    response = requests.get(wikidata_endpoint, params={"query": sparql_query, "format": "json"}, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        return data["results"]["bindings"]
    else:
        print("Error:", response.status_code)
        return None

def query_movie_publication(limit_num,offset=0):
    sparql_query = f"""
        SELECT ?movie ?movieLabel ?time ?language ?languageLabel ?director ?directorLabel  
        WHERE
        {{
          ?movie wdt:P31 wd:Q11424.
          ?movie wdt:P577 ?time.
          ?movie wdt:P364 ?language.
          ?movie wdt:P57 ?director.
          FILTER(?language IN (wd:Q1860))
          FILTER(STRSTARTS(STR(?movie), "http://www.wikidata.org/entity/Q"))
          FILTER(STRSTARTS(STR(?language), "http://www.wikidata.org/entity/Q"))
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".}}
        }}
        LIMIT {limit_num}
    """
    # sparql_query = """
    # SELECT ?item ?itemLabel  ?pubdata
    # WHERE
    # {
    #   ?item wdt:P31 wd:Q11424. # Must be a cat
    #   ?item wdt:P577 ?pubdata.
    #   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } # Helps get the label in your language, if not, then en language
    # }
    # LIMIT 10"""
    return run_sparql(sparql_query)




if __name__ == '__main__':
    from tqdm import tqdm
    import json
    from random import shuffle
    from dateutil import parser

    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Movies/wikidata_movies.json'

    r = 'P577'
    relation_title = 'publication_date'
    limit_num = 50000
    triples = query_movie_publication(limit_num)
    outputs = []
    pbar = tqdm(triples)
    prev_movie = None
    prev_result = None
    for triple in pbar:
        movie = triple["movieLabel"]["value"]
        time = parser.isoparse(triple["time"]["value"]).year
        if movie == prev_movie:
            if time not in prev_result["time"]:
                prev_result["time"].append(time)
            continue

        if prev_result is not None:
            pbar.set_description(f"Saved {len(outputs) + 1} samples")
            outputs.append(prev_result)
            with open(result_file, 'a') as f:
                f.write(json.dumps(prev_result) + '\n')

        prev_movie = movie
        prev_result = {
            "movie": movie,
            "time": [time],
            "director": triple["directorLabel"]["value"],
            "info": triple,
        }


    print("Finished Running!")

# Ways to access the attributes of datetime object
# print(f"Year: {parsed_date.year}")
# print(f"Month: {parsed_date.month}")
# print(f"Day: {parsed_date.day}")
# print(f"Hour: {parsed_date.hour}")
# print(f"Minute: {parsed_date.minute}")
# print(f"Second: {parsed_date.second}")
# print(f"Microsecond: {parsed_date.microsecond}")
# print(f"Timezone: {parsed_date.tzinfo}")