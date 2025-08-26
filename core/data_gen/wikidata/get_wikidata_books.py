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

def query_triples(relation_id, limit_num, offset=0):
    sparql_query = """
    SELECT ?headEntity ?tailEntity ?headEntityLabel ?tailEntityLabel
    WHERE {{
      ?headEntity wdt:{} ?tailEntity.
      FILTER(STRSTARTS(STR(?headEntity), "http://www.wikidata.org/entity/Q"))
      FILTER(STRSTARTS(STR(?tailEntity), "http://www.wikidata.org/entity/Q"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {}
    OFFSET {}
    """.format(relation_id, limit_num, offset)
    return return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results']['bindings']


def query_book_author(limit_num,offset=0):
    sparql_query = f"""
        SELECT ?book ?bookLabel ?author ?authorLabel
        # SELECT ?book ?bookLabel  
        WHERE
        {{
          ?book wdt:P31 wd:Q47461344. 
          ?book wdt:P50 ?author. 
          ?author wdt:P27 ?country.
          FILTER(?country IN (wd:Q30, wd:Q145))
          FILTER(STRSTARTS(STR(?book), "http://www.wikidata.org/entity/Q"))
          FILTER(STRSTARTS(STR(?author), "http://www.wikidata.org/entity/Q"))
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".}}
        }}
        LIMIT {limit_num}
    """
    return run_sparql(sparql_query)
    # return return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results']['bindings']

def query_title(entity_id):
    sparql_query = """
    SELECT ?articleTitle
    WHERE {{
      wd:{} rdfs:label ?articleTitle.
      FILTER(LANG(?articleTitle) = "en")
    }}
    """.format(entity_id)
    return return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results']['bindings']

def query_pop(title):
    # headers = {'User-Agent': 'zhuoran.jin@nlpr.ia.ac.cn'}
    headers = {'User-Agent': 'xxx@nlpr.ia.ac.cn'}
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{}/monthly/2022010100/2022013100'
    resp = requests.get(url.format(title), headers=headers)
    if 'items' not in resp.json():
        return -1
    views = resp.json()['items'][0]['views']
    return views


def query_pop_continue(title,pbar,max_count=20):
    pbar.set_description(f"Processing {title}")
    headers = {'User-Agent': 'xxx@nlpr.ia.ac.cn'}
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{}/monthly/2022010100/2022013100'

    num_try = 0
    while True:
        if num_try > max_count:
            pbar.set_description("Reach maximum trying!")
            return -1
        try:
            num_try += 1
            resp = requests.get(url.format(title), headers=headers)
            # Process the response if needed
            # print("Request successful")
        except requests.RequestException as e:
            # Report the error
            pbar.set_description(f"Error: {e}")
            # Sleep for one second before retrying
            time.sleep(3)
            continue

        # Break out of the loop if the request was successful
        if resp.status_code == 200:
            break
        else:
            pbar.set_description(f"Error response code:{resp.status_code}")



    if 'items' not in resp.json():
        return -1
    views = resp.json()['items'][0]['views']
    return views


if __name__ == '__main__':
    from tqdm import tqdm
    import json
    from random import shuffle

    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Books/book_author.json'

    r = 'P50'
    relation_title = 'author'
    limit_num = 5000
    triples = query_book_author(limit_num)
    outputs = []
    shuffle(triples)
    pbar = tqdm(triples)
    for triple in pbar:
        h = triple['book']['value'].split('/')[-1]
        t = triple['author']['value'].split('/')[-1]
        h_title = triple["bookLabel"]["value"]
        t_title = triple["authorLabel"]["value"]
        if len(h_title) == 0 or len(t_title) == 0:
            continue
        h_pop = query_pop_continue(h_title,pbar)
        t_pop = query_pop_continue(t_title,pbar)
        if h_pop == -1 or t_pop == -1:
            continue
        outputs.append(
            {'sample_id':f"{h}_{r}_{t}",'subject_id': h, 'subject_title': h_title, 'subject_pop': h_pop,
             'relation_id': r, 'relation_title': relation_title,
             'object_id': t, 'object_title': t_title, 'object_pop': t_pop})
        with open(result_file, 'a') as f:
            f.write(json.dumps(outputs[-1]) + '\n')
    print("Finished Running!")
