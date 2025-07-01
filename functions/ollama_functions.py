from ollama import chat

#OLLAMA_HOST=127.0.0.1:11435 ollama serve

def ollama_german_ner(text, model_name="llama3.2:3b"):
    """
    Perform German Named Entity Recognition (NER) using a LLaMA-based model
    via the Ollama Python client and return the results in a custom text format.

    :param text: The German text to analyze for named entities.
    :param model_name: The name/path of the model to use (default: "llama3.2:1b").
    :return: A single string with the extracted entities in the desired format.
    """


    prompt = f"""
    You are a Named Entity Recognition (NER) system.  
    Extract the named entities :
    - Person: a person's name
    - Location: a place name or location
    from the following German text and return the result in exactly the following format:

    - Entities: <Entity Name>, <more entities if necessary>

    ⚠️ Important Rules:
    1. **Only return entities that actually appear in the text**.
    2. **Do not invent new entities**.
    3. If there are no entities, write “None”.
    4. The format must be strictly followed.

    Text:
    \"\"\"{text}\"\"\"
    """

    # Send the prompt to the model using the Ollama Python client.
    response = chat(
        model=model_name,
        options={'temperature': 0},
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    # The LLM’s text is in response['message']['content'].
    model_output = response['message']['content'].strip()
    return model_output


if __name__ == "__main__":
    # Test the function with a sample text.
    text = "Die Hauptstadt von Deutschland ist Berlin."
    entities = ollama_german_ner(text)
    print(entities)
