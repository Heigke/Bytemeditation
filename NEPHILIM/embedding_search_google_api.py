import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from googlesearch import search
import numpy as np
from sentence_transformers import SentenceTransformer
from operator import itemgetter
from googleapiclient.discovery import build
from tqdm import tqdm
# Replace with your own API key and search engine ID
my_api_key = ""
my_cse_id = ""

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']




# Load a pre-trained model from Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')  # change this line if you use a different model

def compare_embeddings(embedding1, embedding2):
    # Compute the cosine similarity between the embeddings
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    return similarity

def scrape_webpage(url,q,s, T=10,max_links=10, timeout=20):
    max_sim_query = {'paragraph': '', 'similarity': -np.inf}
    max_sim_string = {'paragraph': '', 'similarity': -np.inf}
    another_string = s  # Replace this with the string you want to compare
    if T == 0 or max_links == 0:
        return max_sim_query, max_sim_string
    try:
        # Send a GET request to the webpage with a timeout
        request = requests.get(url, timeout=timeout)
    except (requests.exceptions.RequestException, requests.exceptions.Timeout) as err:
        print ("Request error: ", err)
        return max_sim_query, max_sim_string
    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(request.text, 'html.parser')

    # Generate the embeddings for the query and another string
    emb_query = model.encode(q).flatten()
    emb_another_string = model.encode(another_string).flatten()

    # Store the content of all <p> tags:
    paragraphs = [paragraph.get_text() for paragraph in soup.find_all('p')]
    for paragraph in tqdm(paragraphs,desc="Paragraphs"):
        # Compute embeddings and similarity for each paragraph
        emb_paragraph = model.encode(paragraph).flatten()
        query_sim = compare_embeddings(emb_query, emb_paragraph)
        another_string_sim = compare_embeddings(emb_another_string, emb_paragraph)
        
        if query_sim > max_sim_query['similarity']:
            max_sim_query = {'paragraph': paragraph, 'similarity': query_sim}

        if another_string_sim > max_sim_string['similarity']:
            max_sim_string = {'paragraph': paragraph, 'similarity': another_string_sim}

    # Iterate over all URLs found within a page’s <a> tags:
    for link in tqdm(soup.find_all('a'),desc="find a"):
        href = link.get('href')
        # Ensure the URL is absolute:
        abs_url = urljoin(url, href)
        # Recursively scrape the linked page:
        query_result, string_result = scrape_webpage(abs_url,q,s, T-1)
        
        if query_result['similarity'] > max_sim_query['similarity']:
            max_sim_query = query_result
        
        if string_result['similarity'] > max_sim_string['similarity']:
            max_sim_string = string_result

    return max_sim_query, max_sim_string








def get_web_result(q1,q2,T=1):
#	try:

		# Initialize lists to store the results
		query_results = []
		string_results = []
		final_results=[]
	#	q1="Will AI take over the world?"
	#	q2="There is a big chance that AI will take over the world."
		# Run the search

		results = google_search(q1, my_api_key, my_cse_id, num=10)

		for result in tqdm(results,desc="main result loop"):
			if "link" not in result:
				continue
			print(result['link'], result['snippet'])


			#		for v in search(q1, advanced=True):
			#		    print(v.url)
			#		    print(v.title)
			#		    print(v.description)
			query_result, string_result = scrape_webpage(result['link'], q1, q2, T)

			# Calculate the average similarity and add the result to the list
			avg_similarity = (query_result['similarity'] + string_result['similarity']) / 2
			final_results.append({'paragraph': query_result['paragraph'], 'avg_similarity': avg_similarity})

		# Sort the results by average similarity
		final_results.sort(key=itemgetter('avg_similarity'), reverse=True)

		# Print the best result
		print("Closest paragraph to both strings:")
		print("Paragraph: ", final_results[0]['paragraph'])
		print("Average similarity: ", final_results[0]['avg_similarity'])
		return final_results[0]['paragraph']
#	except Exception as e:
#		print(e)
#		return ""

#a=get_web_result("How does other people perceive their world?","Wildly different")
#print(a)
