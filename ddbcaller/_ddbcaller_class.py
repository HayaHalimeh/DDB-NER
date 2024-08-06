
import requests
from requests.structures import CaseInsensitiveDict
from typing import List, Mapping, Union
from functions.utils import *



class DDBCaller(object):
    """docstring for DDBCaller"""

    def __init__(self, oauth_consumer_key):
        super(DDBCaller, self).__init__()

        self.oauth_consumer_key = oauth_consumer_key

        self.headers = CaseInsensitiveDict()
        self.headers["accept"] = "application/json"
        self.headers["Content-Type"] = "application/json"

        self.url = 'https://api.deutsche-digitale-bibliothek.de/search'


    def get_organisation(self, organisation: str, arg: dict) -> List[Mapping[str, Union[str, List[str]]]]:

        """
        Get a list of organisations from the DDB API
        :param organisation: The name of the organisation
        :return: A list of organisations
        """

        params = {
            'query': organisation,
            'rows': 1000,
            'offset': 0,
            'sort': 'RELEVANCE',
            'oauth_consumer_key': self.oauth_consumer_key
        }

        params.update(arg)

        try:

            resp = requests.get(self.url + '/organization', params=params, headers=self.headers)
            if validate_response(resp):
                data = resp.json()
                return data
            else:
                print(f"Error: Invalid response '{str(resp)}'")
                return None
            
        except requests.exceptions.RequestException as e:
            (f"Error: Unable to access '{resp}': {str(e)}")
            return None


    def get_person(self, person: str, arg: dict) -> List[Mapping[str, Union[str, List[str]]]]:

        """
        Get a list of persons from the DDB API
        :param person: The name of the person
        :return: A list of persons
        """

        params = {
            'query': person,
            'rows': 1000,
            'offset': 0,
            'sort': 'RELEVANCE',
            'oauth_consumer_key': self.oauth_consumer_key
        }

        params.update(arg)

        try:

            resp = requests.get(self.url + '/person', params=params, headers=self.headers)
            if validate_response(resp):
                data = resp.json()
                return data
            else:
                print(f"Error: Invalid response '{str(resp)}'")
                return None
            
        except requests.exceptions.RequestException as e:
            (f"Error: Unable to access '{resp}': {str(e)}")
            return None
    

    def get_query(self, query: str, arg: dict) -> List[Mapping[str, Union[str, List[str]]]]:

        """
        Get a list of organisations from the DDB API
        :param organisation: The name of the organisation
        :return: A list of organisations
        """

        params = {
            'query': query,
            'rows': 1000,
            'offset': 0,
            'sort': 'RELEVANCE',
            'oauth_consumer_key': self.oauth_consumer_key
        }

        params.update(arg)
        try:

            resp = requests.get(self.url , params=params, headers=self.headers)
            if validate_response(resp):
                data = resp.json()
                return data
            else:
                print(f"Error: Invalid response '{str(resp)}'")
                return None
                
        except requests.exceptions.RequestException as e:
            (f"Error: Unable to access '{resp}': {str(e)}")
            return None

