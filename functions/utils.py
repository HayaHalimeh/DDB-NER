import requests 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from german_ner.GermanNER import GermanNerModel
from typing import List, Mapping, Union




def read_access_token_from_file(token_file_path):
    """
    Read an access token from a text file.

    Args:
        token_file_path (str): The path to the text file containing the access token.

    Returns:
        str or None: The access token read from the file, or None if the file is not found.

    This function reads the content of a text file at the specified 'token_file_path'.
    It expects the file to contain a single access token and returns it as a string.
    If the file is not found, it prints an error message and returns None.
    """

    try:
        with open(token_file_path, 'r') as file:
            access_token = file.read().strip()
        return access_token
    except FileNotFoundError:
        print(f"Error: The token file '{token_file_path}' was not found.")
        return None
    


def validate_response(response) -> bool:
    """
    Validate the response object by checking if its string representation equals '<Response [200]>'.

    Args:
        response (requests.Response): The response object from an HTTP request.

    Returns:
        bool: True if the response is '<Response [200]>', otherwise False.
    """
    return str(response) == "<Response [200]>"




def concatenate_entities(entities):

    """
    Processes a list of tuples representing words and their corresponding labels, 
    and concatenates them into sentences based on their labels.

    This function groups words into sentences based on their entity labels. Words with 
    the same label are concatenated into a sentence. A new sentence is started when a 
    word with a different label is encountered. Additionally, words of length 1 or 2 are 
    concatenated without a space, while longer words are concatenated with a space.

    Parameters:
    - entities (list of tuples): Each tuple contains a word and its entity label. 
      The tuple can be in the format (word, label) or (word,) for words without labels.

    Returns:
    - list of str: A list of sentences, each is a string formed by concatenating words 
      based on their labels and lengths.


    Example Usage:
    input_tuples = [('Hans', 'I-PER'), ('Christian', 'I-PER'), ('Andersen', 'I-PER'), 
                    ('Theater', 'I-ORG'), ('Liber', 'I-ORG'), ('i', 'I-ORG'), 
                    ('Hans', 'I-PER'), ('Christian', 'I-PER'), ('Andersen', 'I-PER')]
    sentences = concatenate_entities(input_tuples)
    print("\n".join(sentences))

    This will output:
    Hans Christian Andersen
    Theater Liberi
    Hans Christian Andersen
    
    """

    sentences = []
    current_sentence = []
    labels = []

    for item in entities:
        if len(item) == 2:
            word, label = item
        
        #if empty string, start new string but if starts with ## continue
        if not current_sentence:
            if word.startswith("##"):
                continue

            current_sentence.append((word, label))
            labels.append(label[-3:])

        else:
            if current_sentence[-1][1] == label:
                current_sentence.append((word, label))

            else:
                if word.startswith("##"):
                    current_sentence.append((word, label))

                elif current_sentence[-1][1][-3:] == label[-3:]:

                    if current_sentence[-1][1][0] == 'B' and label[0] == 'I':
                        current_sentence.append((word, label))

                    else:
                        
                        # Process current_sentence and append it to sentences
                        joined_sentence = "".join([word.replace("##", "") if word.startswith("##") else " " + word for word, _ in current_sentence]).strip()
                                                
                        # Append the joined sentence to the list of sentences
                        sentences.append(joined_sentence)

                        # Clear the current sentence
                        current_sentence = [(word, label)]
                        labels.append(label[-3:])                        
                                            

                else:
                    
                    # Process current_sentence and append it to sentences
                    joined_sentence = "".join([word.replace("##", "") if word.startswith("##") else " " + word for word, _ in current_sentence]).strip()
                                        
                    # Append the joined sentence to the list of sentences
                    sentences.append(joined_sentence)
                    
                    # Clear the current sentence
                    current_sentence = [(word, label)]
                    labels.append(label[-3:])
                    
    # Process the last sentence
    if current_sentence:
        joined_sentence = "".join([word.replace("##", "") if word.startswith("##") else " " + word for word, _ in current_sentence]).strip() #word if len(word) in [1, 2] else 
        
        sentences.append(joined_sentence)
    return sentences, labels




def extract_gnd_info(data : dict) -> List[Mapping[str, Union[str, List[str]]]]:
    
    """
    Extract GND (Gemeinsame Normdatei) information from the provided data (supposed to be ddb response json or a snippet of it).

    Args:
        data (dict): A dictionary containing GND information.

    Returns:
        List[Mapping[str, Union[str, List[str]]]]: A list of dictionaries containing extracted GND information.

    The function extracts GND information, including GND ID, label, and additional fields
    based on document type, from the input data. It organizes the extracted information
    into dictionaries and returns them as a list. 
    """

    if data is None:
        return None
    
    elif data['numberOfResults'] == 0:
        return None
    
    else:
        results = data.get("results", [])
        gnd_info = []

        for result in results:
            docs = result.get("docs", [])
            for doc in docs:
                gnd_id = doc.get("id", "")
                label = doc.get("label", "")
                doc_type = doc.get("type", "")

                info = {
                    "gnd_id": gnd_id,
                    "label": label,
                    "type": doc_type,
                }

                if doc_type == "gnd-organization":
                    state_de = doc.get("states_de", [])[0] if doc.get("states_de") else ""
                    city_de = doc.get("city_de", [])[0] if doc.get("city_de") else ""
                    
                    info.update({
                        "state_de": state_de,
                        "city_de": city_de
                    })
                
                    
                elif doc_type == "person":

                    placeOfBirth = doc.get("placeOfBirth", []) if doc.get("placeOfBirth") else ""
                    dateOfBirth_de = doc.get("dateOfBirth_de", []) if doc.get("dateOfBirth_de") else ""
                    professionOrOccupation = doc.get("professionOrOccupation", []) if doc.get("professionOrOccupation") else ""

                    info.update({
                        "dateOfBirth_de": dateOfBirth_de,
                        "placeOfBirth": placeOfBirth,
                        "professionOrOccupation": professionOrOccupation
                    })

                gnd_info.append(info)

        type_name = doc_type if doc_type in ["gnd-organization", "person"] else "unknown"
        
        return {
            "type": type_name,
            "numberOfResults": len(gnd_info),
            "results": gnd_info
        }
        




def get_ddb_url(entry: dict) -> str:
    """
    Return the DDB URL for a given GND ID.

    Args:
        gnd_id (str): The GND ID for which the DDB URL is to be generated.

    Returns:
        str: The DDB URL for the given GND ID.

    This function generates the DDB URL for a given GND ID by appending the ID to the DDB base URL.
    """

    if entry is not None:
        prefix = "https://www.deutsche-digitale-bibliothek.de/"
        

        #entry= #next(iter(entry.values()))
        gnd_id = entry.get("gnd_id")

        if gnd_id:
            #print("entry: ", entry)
            
            
            if entry["type"] == "person":
                if "gnd/" in gnd_id:
                    url = prefix + "person/" + gnd_id[gnd_id.index("gnd/"):]
                else:
                    url = prefix + "person/" + gnd_id
                    
            elif entry["type"] == "gnd-organization" or entry["type"] == "ddb-institution":
                if "gnd/" in gnd_id:
                    url = prefix + "organization/" + gnd_id[gnd_id.index("gnd/"):]
                else:
                    url = prefix + "organization/" + gnd_id

            else:                    
                url = prefix + "item/" + gnd_id

            try:
                # Send a GET request to the URL
                response = requests.get(url)
                if validate_response(response):
                    return url
                
            except requests.exceptions.RequestException as e:
                # Handle network-related errors
                print(f"Error: Unable to access '{url}': {str(e)}")
                
    return None




def compute_jaccard_distance(query: str, text: str) -> float:
    """
    Compute the Jaccard distance between two strings after normalizing.

    Args:
        query (str): The first input string.
        text (str): The second input string.

    Returns:
        float: The Jaccard distance between the two input strings.

    This function computes the Jaccard distance between two input strings.

    The Jaccard distance is calculated as 1 - (|A ∩ B| / |A ∪ B|), where A and B are the sets of words in the two strings.
    """

    import re
    # Tokenize the strings into sets of words, removing punctuation and converting to lowercase
    set1 = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    set2 = set(re.sub(r"[^\w\s]", "", text.lower()).split())

    # Calculate intersection and union
    intersection = set1 & set2
    union = set1 | set2

    # Calculate Jaccard index and distance
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 1
    return 1 - jaccard_index




def sorted_jaccard_distance(query: str, gnd_info: dict) -> dict:
    """
    Compute Jaccard distance for a query and sort the results.

    Args:
        query (str): The query for which Jaccard distance is to be computed.
        gnd_info (dict): A dictionary containing GND information including labels.

    Returns:
        dict: A dictionary containing the sorted Jaccard distances for the query.

    This function computes the Jaccard distance between the query and labels in the provided GND information.
    It then sorts the results based on the Jaccard distance and returns a dictionary with the sorted results.
    """

    if not query or not gnd_info or gnd_info is None:
        return None
    
    results = [
        {
            "label": entry["label"],
            "gnd_id": entry.get("gnd_id") if entry.get("gnd_id") else "N/A",
            "Jaccard-Distance": compute_jaccard_distance(query, entry["label"]),
            "type": gnd_info["type"] if gnd_info["type"] in ["gnd-organization", "person"] else "unknown"

        }
        for entry in gnd_info['results']
    ]
    
    sorted_results = sorted(results, key=lambda x: x["Jaccard-Distance"])
    results_dict = {f"Result_{str(i)}": sorted_results[i] for i in range(len(sorted_results))}

    return results_dict




def find_jaccard_best_match(query: str, gnd_info: dict) -> str:

    """
    Return the best match for the given query.

    Args:
        query (str): The query for which a best match is to be determined.
        gnd_info (dict): A dictionary containing sorted gnd info based on jaccard similarity.

    Returns:
        str: The URL of the best match or None if no match is found.

    The function returns the DDB URL of the best match based on the document type in the GND information.
    If no match is found, it returns None.
    """
    if gnd_info is None or query is None:
        return None

    first_key = next(iter(gnd_info))
    first_value = gnd_info[first_key]
    if first_value.get("Jaccard-Distance") == 0:
            return get_ddb_url(first_value)
    else:
        return None
    
    




def find_cosine_best_match(ner_model, query: str, gnd_info: dict) -> str:

    """
    Return the best match for the given query.

    Args:
        query (str): The query for which a best match is to be determined.
        gnd_info (dict): A dictionary containing sorted gnd info based on jaccard similarity.

    Returns:
        str: The URL of the best match or None if no match is found.

    The function returns the DDB URL of the best match based on the document type in the GND information.
    If no match is found, it returns None.
    """
    if gnd_info is None or query is None:
        return None
    
    list_of_results = []
    for result in gnd_info['results']:
        for key, value in result.items():
            if key == 'label':
                list_of_results.append(value)

    if list_of_results == []:
        return None

    
    best_index, max_similarity = ner_model.find_best_cosine_similarity(query, list_of_results)
    print("best_index: ", best_index, "max_similarity: ", max_similarity)
    if best_index is None:
        return None, 0.0
    else:
        return get_ddb_url(gnd_info['results'][best_index]), max_similarity    
        


def compute_sbert_cosine_and_find_best_match(model, reference, candidates):
    """
    Compute cosine similarity between a reference and a list of candidate strings.

    Args:
        model: The sentence-transformers model used to compute the embeddings.
        reference (str): The reference string to compare against.
        candidates (list): A list of candidate strings to compare with the reference.

    Returns:
        str: The candidate string with the highest cosine similarity to the reference.

    """
    #Encode the reference sentence and all candidate sentences
    embeddings = model.encode([reference] + candidates)
    
    # Extract the embedding for the reference sentence
    reference_embedding = embeddings[0]
    
    # Calculate cosine similarities between reference and each candidate
    cos_sims = cosine_similarity([reference_embedding], embeddings[1:])
    
    # Find the index of the highest cosine similarity
    max_index = np.argmax(cos_sims)
    
    # Return the most similar candidate
    return candidates[max_index], cos_sims[0][max_index]
    

