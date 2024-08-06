import torch
from fuzzywuzzy import fuzz
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity



class GermanNerModel:

    """
    A class for initializing and performing Named Entity Recognition (NER) in German using Hugging Face models.

    Args:
        model_name (str): The name or path of the pre-trained NER model.

    Attributes:
        model_name (str): The name or path of the pre-trained NER model.
        tokenizer (AutoTokenizer): The tokenizer for tokenizing input text.
        ner_model (AutoModelForTokenClassification): The NER model for entity recognition.
    """

    def __init__(self, model_name):
        super(GermanNerModel, self).__init__()

        """
        Initializes a GermanNerModel instance.

        Args:
            model_name (str): The name or path of the pre-trained NER model.
        """

        self.model_name = model_name
        self.tokenizer = None
        self.ner_model = None
        self.load_model()


    def load_model(self):
        """
        Loads the pre-trained NER model and tokenizer from Hugging Face.
        """
        
        try:
            # Initialize the tokenizer with the access token
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, )
            
            # Initialize the NER model with the access token
            self.ner_model = AutoModelForTokenClassification.from_pretrained(self.model_name, )
        except Exception as e:
            print(f"Error: Unable to initialize German NER model with token - {str(e)}")
            self.tokenizer = None
            self.ner_model = None

        

    def perform_ner(self, label_suffixes, text):

        """
        Performs Named Entity Recognition (NER) on the input `text` and filters entities with labels ending in
        the specified `label_suffixes`.

        Args:
            label_suffixes (list): A list of label suffixes (e.g., ["-PER", "-ORG"]) to filter entities by.
            text (str): The input text to perform NER on.

        Returns:
            list: A list of filtered (entity, label) tuples.

        Raises:
            Exception: If there is an error during NER prediction.

        Example Usage:
            ner_model = GermanNerModel("xlm-roberta-large-finetuned-conll03-german")
            entities = ner_model.perform_ner(["-PER", "-ORG"], "Angela Merkel ist die Bundeskanzlerin von Deutschland.")
            print(entities)
        """

        try:
            filtered_entities = []
            if self.tokenizer and self.ner_model:

                inputs = self.tokenizer(text, return_tensors="pt")
                outputs = self.ner_model(**inputs).logits
                predictions = torch.argmax(outputs, dim=2)
          
                entities = []
                for idx, pred in enumerate(predictions[0].numpy()):
                  
                    if pred != 0:  # 0 Tag for 'not an entity'
                        entity = self.tokenizer.decode(inputs.input_ids[0][idx])
                        entities.append((entity, self.ner_model.config.id2label[pred]))
                        
                filtered_entities = [entity for entity in entities if any(entity[1].endswith(prefix) for prefix in label_suffixes)]
                
            
            else:
                print("Tokenizer or NER model is not initialized.")
            
            return filtered_entities
            
        except Exception as e:
            print(f"Error: Unable to perform NER - {str(e)}")
            return []
        

    def entities_to_vec(self, entities):
        """
        Converts a list of entities to a single vector by averaging the embeddings of the entities.

        Args: 
            entities (list): A list of entities recognized by the NER model.

        Returns:
            torch.Tensor: A single vector representing the entities.

        """
        vecs = []
        for entity in entities:
            # Extract the token IDs and create a tensor
            token_ids = self.tokenizer.encode(entity[0], add_special_tokens=False)
            tokens_tensor = torch.tensor([token_ids])

            # Get embeddings and take the mean
            with torch.no_grad():
                outputs = self.ner_model(tokens_tensor)
                embeddings = outputs[0].mean(dim=1)
            vecs.append(embeddings)

        if vecs:
            # Stack and average the vectors
            return torch.stack(vecs).mean(dim=0)
        else:
            # Return a zero vector if no entities were found
            return torch.zeros(self.ner_model.config.hidden_size)



    def find_best_cosine_similarity(self, query, candidates):
        """
        Calculate the cosine similarity between the query and a list of candidate texts, using NER-identified entities
        and fuzzy matching for additional evaluation.

        Args:
            query (str): The query text.
            candidates (list): A list of candidate texts to compare with the query.

        Returns:
            tuple: A tuple containing the index of the best match from the candidate texts and the combined similarity score.

        Example Usage:
            ner_model = GermanNerModel()
            best_index, similarity = ner_model.find_best_cosine_similarity(
                "Angela Merkel ist die Bundeskanzlerin von Deutschland.",
                ["Merkel ist die Kanzlerin von Deutschland.", "Die Kanzlerin von Deutschland ist Angela Merkel."]
            )
            print(best_index, similarity)
        """

        # Perform NER on the query
        entities_query = self.perform_ner(["-PER", "-ORG"], query)
        vec_query = self.entities_to_vec(entities_query)

        if len(entities_query) == 0:
            return None, 0.0

        max_combined_score = 0.0
        best_index = None

        for index, entry in enumerate(candidates):

            # Perform NER on the candidate entry
            entities_candidate = self.perform_ner(["-PER", "-ORG"], entry)
            vec_candidate = self.entities_to_vec(entities_candidate)

            # Check if the vectors have the same shape
            if vec_query.shape == vec_candidate.shape:
                cos_sim = cosine_similarity(vec_query.reshape(1, -1), vec_candidate.reshape(1, -1))[0][0]

                # Calculate the fuzzy matching score
                fuzzy_score = fuzz.ratio(query, entry) / 100.0

                # Combine the scores with a weighted average
                combined_score = (0.7 * cos_sim) + (0.3 * fuzzy_score)

                # Update the best match if the current combined score is greater than the maximum found so far
                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    best_index = index

        return best_index, max_combined_score

    

#### absolute tokens
# Path: german_ner/GermanNER.py
    '''
    def find_best_cosine_similarity(self, query, candidates):
        """
        Calculate the cosine similarity between two texts based on the entities recognized by the NER model.

        Args:
            s1 (str): The first input text.
            candidates (list): A list of candidate texts to compare with the first text.

        Returns:
            tuple: A tuple containing the best match from the candidate texts and the cosine similarity score.

        Example Usage:
            ner_model = GermanNerModel("xlm-roberta-large-finetuned-conll03-german")
            best_match, similarity = ner_model.find_best_cosine_similarity("Angela Merkel ist die Bundeskanzlerin von Deutschland.", ["Merkel ist die Kanzlerin von Deutschland.", "Die Kanzlerin von Deutschland ist Angela Merkel."])
            print(best_match, similarity)
        """

        entities_query = self.perform_ner(["-PER", "-ORG"], query)
        vec_query= self.entities_to_vec(entities_query)
        
        if len(entities_query) == 0:
            return None, 0.0
        
        max_similarity = 0.0
        #best_match = None
        best_index = None
        
        for index, entry in enumerate(candidates):
            print('index, entry ', index, entry )
            entities_candidate = self.perform_ner(["-PER", "-ORG"], entry)
            vec_candidate= self.entities_to_vec(entities_candidate)
            if vec_query.shape == vec_candidate.shape:
                cos_sim = cosine_similarity(vec_query.reshape(1, -1), vec_candidate.reshape(1, -1))[0][0]
                print('cos_sim', cos_sim)
                if cos_sim > max_similarity:
                    max_similarity = cos_sim
                    #best_match = entry
                    best_index = index

        print('best_index, max_similarity', best_index, max_similarity)
        return best_index, max_similarity


    def calculate_cosine_similarity_ner(self, text1, text2):
        """
        Calculate the cosine similarity between two texts based on the entities recognized by the NER model.

        Args:
            text1 (str): The first input text.
            text2 (str): The second input text.

        Returns:
            float: The cosine similarity between the two texts based on the entities recognized by the NER model.

        Example Usage:
            ner_model = GermanNerModel("xlm-roberta-large-finetuned-conll03-german")
            similarity = ner_model.calculate_cosine_similarity_ner("Angela Merkel ist die Bundeskanzlerin von Deutschland.", "Merkel ist die Kanzlerin von Deutschland.")
            print(similarity)
        """

        try:
            entities1 = self.perform_ner(["-PER", "-ORG"], text1)
            entities2 = self.perform_ner(["-PER", "-ORG"], text2)

            if entities1 and entities2:
                entity_set1 = set([entity[0] for entity in entities1])
                entity_set2 = set([entity[0] for entity in entities2])

                common_entities = entity_set1.intersection(entity_set2)
                if len(common_entities) == 0:
                    return 0.0

                return len(common_entities) / (len(entity_set1) * len(entity_set2)) ** 0.5

            return 0.0

        except Exception as e:
            print(f"Error: Unable to calculate cosine similarity - {str(e)}")
            return 0.0

    '''
