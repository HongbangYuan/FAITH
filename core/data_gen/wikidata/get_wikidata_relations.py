from dotenv import load_dotenv

load_dotenv()
from qwikidata.sparql import return_sparql_query_results
import requests

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


def query_film_director(limit_num,offset=0):
    sparql_query = f"""
        SELECT ?work ?director  ?workLabel ?directorLabel
        WHERE
        {{
          ?work wdt:P31 wd:Q11424. 
          ?work wdt:P57 ?director. 
          FILTER(STRSTARTS(STR(?work), "http://www.wikidata.org/entity/Q"))
          FILTER(STRSTARTS(STR(?director), "http://www.wikidata.org/entity/Q"))
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }}
        }}
        LIMIT {limit_num}
    """
    return return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results']['bindings']

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
    headers = {'User-Agent': 'zhuoran.jin@nlpr.ia.ac.cn'}
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{}/monthly/2022010100/2022013100'
    resp = requests.get(url.format(title), headers=headers)
    if 'items' not in resp.json():
        return -1
    views = resp.json()['items'][0]['views']
    return views



if __name__ == '__main__':
    relation_id = 'P57'
    relation_name = 'director'
    limit_num = 30
    # triples = query_triples(relation_id,limit_num)
    triples = query_film_director(limit_num)
    output = []
    for triple in triples[:3]:
        h = triple['work']['value'].split('/')[-1]
        t = triple['director']['value'].split('/')[-1]
        # if query_ntail(h, r) == 1:
        h_title = query_title(h)
        t_title = query_title(t)
        if len(h_title) == 0 or len(t_title) == 0:
            continue
        h_title = h_title[0]['articleTitle']['value']
        t_title = t_title[0]['articleTitle']['value']
        h_pop = query_pop(h_title)
        t_pop = query_pop(t_title)
        # h_alias = query_alias(h)
        # t_alias = query_alias(t)
        if h_pop == -1 or t_pop == -1:
            continue
        # output.append(
        #     {'subject_id': h, 'subject_title': h_title, 'subject_alias': h_alias, 'subject_pop': h_pop,
        #      'relation_id': r, 'relation_title': 'country',
        #      'object_id': t, 'object_title': t_title, 'object_alias': t_alias, 'object_pop': t_pop})
        # output.append(
        #     {'subject_id': h, 'subject_title': h_title, 'subject_pop': h_pop,
        #      'relation_id': r, 'relation_title': relation_title,
        #      'object_id': t, 'object_title': t_title, 'object_pop': t_pop})
        print(h_title, h_pop, t_title, t_pop)
