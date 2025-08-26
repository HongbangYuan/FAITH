import requests
from dotenv import load_dotenv

load_dotenv()


def query_books_and_authors():
    # Define the SPARQL query to retrieve 100 kinds of books and their famous authors
    sparql_query = """
    SELECT ?book ?bookLabel ?author ?authorLabel
    WHERE {
      ?book wdt:P31 wd:Q571.  # Instance of book
      ?book wdt:P50 ?author.  # Author relationship
      # ?author wdt:P19 wd:Q30.  # Author born in USA or UK
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 100
    """

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

        # Extract relevant information from the response
        results = []
        for item in data["results"]["bindings"]:
            result = {
                "book_id": item["book"]["value"].split("/")[-1],
                "book_label": item["bookLabel"]["value"],
                "author_id": item["author"]["value"].split("/")[-1],
                "author_label": item["authorLabel"]["value"]
            }
            results.append(result)

        return results
    else:
        print("Error:", response.status_code)
        return None

def query_books_of_one_author():
    # Define the SPARQL query to retrieve 100 kinds of books and their famous authors
    sparql_query = """
    SELECT ?book ?bookLabel  
    WHERE {
      ?book wdt:P50 wd:Q35610.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 100
    """

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

        # Extract relevant information from the response
        results = []
        for item in data["results"]["bindings"]:
            result = {
                "book_id": item["book"]["value"].split("/")[-1],
                "book_label": item["bookLabel"]["value"],
            }
            results.append(result)

        return results
    else:
        print("Error:", response.status_code)
        return None

def query_wikidata_relations():
    # Define the SPARQL query to retrieve ten relations
    sparql_query = """
    SELECT ?relation ?relationLabel
    WHERE {
      ?relation a wikibase:Property.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 100
    """

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

        # Extract relevant information from the response
        results = []
        for item in data["results"]["bindings"]:
            result = {
                "relation_id": item["relation"]["value"].split("/")[-1],
                "relation_label": item["relationLabel"]["value"]
            }
            results.append(result)

        return results
    else:
        print("Error:", response.status_code)
        return None

if __name__ == '__main__':
    # # Example: Query Wikidata for ten relations
    # results = query_wikidata_relations()
    #
    # # Print the results
    # if results:
    #     for result in results:
    #         print(f"{result['relation_label']} (P{result['relation_id']})")
    # else:
    #     print("Query failed.")


    # Example: Query Wikidata for 100 kinds of books and their famous authors
    results = query_books_of_one_author()

    # # Print the results
    # if results:
    #     for result in results:
    #         print(f"Book: {result['book_label']} (Q{result['book_id']}) | Author: {result['author_label']} (Q{result['author_id']})")
    # else:
    #     print("Query failed.")