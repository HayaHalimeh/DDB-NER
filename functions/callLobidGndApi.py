
import requests
from requests.structures import CaseInsensitiveDict
import pprint as pprint
import typing
from functions.utils import *

#See https://lobid.org/gnd/api for more information on the API


def get_lodib_data(query) -> typing.Dict[str, str]:
    """
    Get data from the lobid API.
    Args:
        query (str): The query to search for in the lobid API.
        Returns:
        typing.Dict[str, str]: The data returned from the lobid API.
    """

    headers = CaseInsensitiveDict()
    headers["accept"] = "application/json"
    headers["Content-Type"] = "application/json"

    url = f'https://lobid.org/gnd/search?q={query}&format=json'

    resp = requests.get(url, headers=headers)
    if validate_response(resp):
        data = resp.json()
        return data
    else:
        return None


  
def find_collection_idUrl(data, ddb_name) -> typing.Union[str, None]:
    """
    Find the ID URL for collection DDB in the lobid data.
    Args:
        data (typing.Dict[str, str]): The lobid data to search in.
        ddb_name (str): The name of the collection DDB.
        Returns:
        typing.Union[str, None]: The ID URL for the collection DDB if found, else None.
    """
    if data is None:
        return None
    
    else:
        for item in data:
            if 'collection' in item and 'name' in item['collection'] and item['collection']['name'] == ddb_name:
                return item['id']
        return None 


   
def match_ddb_url(data, ddb_name, ddb_url) -> bool:
    """
    Match the DDB URL in the lobid data.
    Args:
        data (typing.Dict[str, str]): The lobid data to search in.
        ddb_name (str): The name of DDB.
        ddb_url (str): The URL of the DDB to match agaiinst.
        Returns:
        bool: True if the DDB URL is found in the lobid data, else False.
    """

    #sameAs = data['member'][0]['sameAs']
    if data['member'] == []:
        return False
    
    ddb_id_url = find_collection_idUrl(data['member'][0]['sameAs'], ddb_name)

    if ddb_id_url == ddb_url:
        return True
    else:
        return False 
            


def match_names(data, entity_name, ner_model) -> bool: 
    """
    Match the entity name in the lobid data with entity name and perform cosine sim if necessary.
    Args:
        data (typing.Dict[str, str]): The lobid data to search in.
        entity_name (str): The name of the entity to match against.
        Returns:
        bool: True if the entity name is found in the lobid data, else False.
        
    """


    min_simalarity = 0.9

    if data['member'] == []:
        return False

    if data['member'][0]['preferredName'] is not None:
        preferredName = data['member'][0]['preferredName']

        if entity_name in [preferredName]:
            return True
        
        else:
            _, max_similarity = ner_model.find_best_cosine_similarity(entity_name, [preferredName])
            if max_similarity >=  min_simalarity:
                return max_similarity
            else:
                return False
    
     
    elif data['member'][0]['preferredNameEntityForThePerson'] is not None:
        if 'forename' and 'surname' in data['member'][0]['preferredNameEntityForThePerson'].keys():
            temp = data['member'][0]['preferredNameEntityForThePerson']['forename'][0] + " " + data['member'][0]['preferredNameEntityForThePerson']['surname'][0] 
            if entity_name in temp:
                return True
            
        
            
    elif data['member'][0]['variantName'] is not None:
        variantName = data['member'][0]['variantName']
        if entity_name in variantName:
            return True
        else:
            _, max_similarity = ner_model.find_best_cosine_similarity(entity_name, variantName)
            if max_similarity >=  min_simalarity:
                return max_similarity
            else:
                return False
    else:
        return False



def match_entity(data, ddb_name, ddb_url, entity_name, ner_model) -> str:
    """
    Match the entity in the lobid data.
    Args:
        data (typing.Dict[str, str]): The lobid data to search in.
        ddb_name (str): The name of DDB.
        ddb_url (str): The URL of the DDB to match agaiinst.
        entity_name (str): The name of the entity to match against.
        Returns:
        str: A message indicating the match found.
    """
    if match_names(data, entity_name):
        return f" {entity_name}"
    elif match_ddb_url(data, ddb_name, ddb_url):
        return f" {ddb_url}"
    
    else:
        return None