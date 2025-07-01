import os
os.environ['dir'] = "./" 
os.chdir(os.environ['dir'])


import streamlit as st
import time
from german_ner.GermanNER import GermanNerModel
from functions.utils import *
from ddbcaller._ddbcaller_class import DDBCaller
from functions.callLobidGndApi import *
from functions.utils import *
from sentence_transformers import SentenceTransformer
from PIL import Image


print(os.getcwd())
os.environ['DDB_API_KEY']= read_access_token_from_file(token_file_path= "./ddb_access_token.txt")
ddb = DDBCaller(os.environ['DDB_API_KEY'])

model_name = "mschiesser/ner-bert-german" 

label_suffixes = ["-PER",  "-ORG"]
ner_model  = GermanNerModel(model_name)
ddb_name = 'Deutsche Digitale Bibliothek'
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")



#App Functions
def make_background_transparent(image_path, output_path, target_color=(255, 255, 255, 255), tolerance=0):
    """
    Make the background of an image transparent with a tolerance for color matching.

    :param image_path: Path to the input image.
    :param output_path: Path where the output image with transparent background will be saved.
    :param target_color: The color to be made transparent. Default is white (255, 255, 255).
    :param tolerance: Color matching tolerance. Increases the range of colors considered "matching".
    """
    image = Image.open(image_path)
    image = image.convert("RGBA")

    newData = []
    for item in image.getdata():
        # Calculate if the current pixel falls within the target_color plus tolerance
        if all(abs(item[i] - target_color[i]) <= tolerance for i in range(3)):
            newData.append((255, 255, 255, 0))  
        else:
            newData.append(item)

    image.putdata(newData)
    image.save(output_path, "PNG")


# Preprocessing functions
def normalize_text(text: str ) -> str:
    import re
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# Extract the 'id' from the given nested data structure if only one results returned
def extract_id(data):
    """
    Extract the 'id' from the given nested data structure.

    Args:
        data (dict): The input data structure containing nested elements.

    Returns:
        str: The extracted 'id', or None if not found.
    """
    try:
        # Access the 'id' in the nested structure
        return data["results"][0]["docs"][0]["id"].split("/")[-1]
    except (IndexError, KeyError) as e:
        # Handle cases where the structure might not be as expected
        print(f"Error accessing ID: {e}")
        return None
 

# DNB Function
def match_against_dnb(ddb_url, entity_name, ner_model):
    """
    Match the entity name in the lobid data with entity name and perform cosine sim if necessary.
    Args:
        data (typing.Dict[str, str]): The lobid data to search in.
        entity_name (str): The name of the entity to match against.
        Returns:
        bool: True if the entity name is found in the lobid data, else False.
        
    """

    if isinstance(ddb_url, str) == True:
        query = ddb_url.split("/")[-1]
        url = f'https://d-nb.info/gnd/{query}/about'

        match = match_names(get_lodib_data(query), entity_name, ner_model)    

        if match is not None:
            st.markdown(" 🔍 Validate result by matching detected entity with the Deutsche Nationalbibliothek ")
            st.markdown(f""" 
            <li> Result verified from the Deutsche Nationalbibliothek: <a href="{url}" target="_blank">{url}</a></li>
            """, unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)
        


def process_entities_by_type(entities, labels, label_type, similarity_threshold, profession=None, city=None, one_name_person=False):

    """
    Process extracted entities of a specific type (PER or ORG), match them with the DDB, and perform similarity checks.
    """
    ddb_url = None
    already_seen = list()

    if label_type ==  "PER":
        input_type = "person"
    else:
        input_type = "location"

    for entity, label in zip(entities, labels):
        
        print("-"*25)
        print("entity, label", entity, label)

        ddb_url= None 

        if label != label_type:
            continue


        if label == "PER" and len(entity.split()) < 2 and one_name_person == False:
            st.write(f" ➡️ Entity {entity} not complete, Skip entity!")
            # Custom styled horizontal line
            st.markdown("""
            <style>
            .divider {
                border-top: 2px solid #bbb;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            </style>
            <div class="divider"></div>
            """, unsafe_allow_html=True)
            continue
        if entity in already_seen:
            continue

        already_seen.append(entity)
        st.write(f" ➡️ Entity {entity}: Entity Type {input_type.capitalize() } detected")

        with st.spinner(f" 🔄 Searching for {entity} in DDB ..."):
        
            # Perform query based on entity type
            if label == "PER":
                data = ddb.get_person(entity, {})
            elif label == "ORG":
                data = ddb.get_organisation(entity, {})
                
            if data is None or data['numberOfResults'] == 0:
                st.markdown(
                f"* No match found for **{entity}: {input_type.capitalize()}**.",
                unsafe_allow_html=True
            )

            else:
                gnd_info = extract_gnd_info(data)
                print(gnd_info)

                if label == "PER" and len(profession) > 0:
                    gnd_info = filter_by_profession(embedding_model, gnd_info, profession, threshold=0.7)

                    if gnd_info['numberOfResults'] == 0:
                        st.markdown(
                        f"* No match found for **{entity}: {input_type.capitalize()}**. Try without the profession filter.",
                        unsafe_allow_html=True
                    )
                        continue


                elif label == "ORG" and len(city) > 0:
                    gnd_info = filter_by_city(gnd_info, city)

                    if gnd_info['numberOfResults'] == 0:
                        st.markdown(
                        f"* No match found for **{entity}: {input_type.capitalize()}**. Try without the city filter.",
                        unsafe_allow_html=True
                    )
                        continue
                
                # Jaccard distance matching
                with st.spinner(' 🔄 Running Jaccard Similarity Search ...'):
                    sorted_results = sorted_jaccard_distance(entity, gnd_info)    
                    ddb_url = find_jaccard_best_match(entity, sorted_results)

                if ddb_url is not None:

                    st.markdown(f""" <br> ✅ Jaccard Search in DDB complete for {entity}""", unsafe_allow_html=True)

                    st.markdown(f"""
                        <li>Best match from DDB: <a href="{ddb_url}" target="_blank">{ddb_url}</a></li>
                        """, unsafe_allow_html=True)

                    st.markdown("</ul></div>", unsafe_allow_html=True)
                        
                else:

                    # Cosine similarity matching
                    with st.spinner('🔄 Running Semantic Similarity Search ...'):                        
                        
                        similarity_threshold = similarity_threshold
                        ddb_url, max_similarity = find_cosine_best_match(ner_model, entity, gnd_info)
                        
                    if max_similarity >= similarity_threshold:

                        st.markdown(f""" <br> ✅ Semantic Search in DDB complete for {entity}
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <li>Best match from DDB: <a href="{ddb_url}" target="_blank">{ddb_url}</a></li>
                        """, unsafe_allow_html=True)

                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                        f"* A total of **{data['numberOfResults']}** possible matches were found in the DDB, but none met the current similarity threshold. Try adjusting the **Semantic Similarity Threshold** to improve results.",
                        unsafe_allow_html=True
                    )
                        ddb_url = None

                if ddb_url is not None:
                    match_against_dnb(ddb_url, entity, ner_model)

        # Custom styled horizontal line
        st.markdown("""
            <style>
            .divider {
                border-top: 2px solid #bbb;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            </style>
            <div class="divider"></div>
            """, unsafe_allow_html=True)
        
    if len(already_seen) == 0 and ddb_url is None:
        st.warning(f"* No {input_type.capitalize()} entities found!")


def main():

    def set_light_theme():
        config_path = "./config.toml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)  

        with open(config_path, "w") as config_file:
            config_file.write(
                """
                [theme]
                base = "light"
                """
            )

    # Set the theme before the app runs
    set_light_theme()

    # Image paths
    logo_path_ddb = "./logo_ddb.png"  
    logo_path_sicp = "./sicp.png"  
    logo_path_da = "./transparent_logo.png"  
    ddb_ner_path = "./ddb_ner_diagram.jpg"  

    # Load the JPEG image
    ddb_ner_image = Image.open(ddb_ner_path)

    # Sidebar content
    st.sidebar.markdown("### Partners")
    st.sidebar.image(logo_path_ddb, use_column_width=True, caption="DDB")
    st.sidebar.image(logo_path_sicp, use_column_width=True, caption="SICP")
    st.sidebar.image(logo_path_da, use_column_width=True, caption="DA Group Upb")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This App")
    
    st.sidebar.markdown("""
    This app allows you to:
    - Extract person and location entities from text.
    - Match entities with databases.
    - View results interactively.
    """)

    # Display the clickable JPEG image in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detailed Logic of the App")
    st.sidebar.markdown("For a detailed view, click the button below.")
    

    # Main content
    st.title("DDB Named Entity Recognition APP")
    if st.sidebar.button("View Full Image"):
        # If the button is clicked, show the image in larger size in the main content area
        st.image(ddb_ner_image, caption="Detailed View", use_column_width=True)
    else:
        # Show the thumbnail version
        st.sidebar.image(ddb_ner_image, caption="Thumbnail", width=150)
    

    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the DDB NER App! This application allows you to extract person and location named entities from text and find matches in the Deutsche Digitale Bibliothek and LOBID databases.
    
    **Instructions**:
    - Select the type of input: Person or Location.
    - Enter the relevant text.
    - Adjust the semantic similarity threshold.
    - Click 'Run Command' to perform the analysis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Selection for input type
    input_type = st.radio("Select the type of input you want to provide:", ("Person", "Location"))

    # Text input based on selected type
    user_input = ""

    
    if input_type == "Person":
        user_input = st.text_input("Enter the name of the person you want to search for:")
        user_input = clean_html_text(user_input)
        user_input = f"{user_input}, ist bekannt" if user_input else ""
        profession =  st.text_input("Enter the name of the professsion you want to restrict the results to:")

        one_name_person = st.checkbox("One Name Person")
        spacy_filtering = st.checkbox("Use Spacy Filtering")
        


    elif input_type == "Location":
        user_input = st.text_input("Enter the name of the location you want to search for:")
        user_input = clean_html_text(user_input)
        user_input = f"Die {user_input} ist ein bekannter Ort" if user_input else ""
        city =  st.text_input("Enter the name of the city you want to restrict the results to:")

        # Add the similarity threshold slider under the text input
    similarity_threshold = st.slider(
        "Semantic Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.01,
        help="Adjust the threshold for semantic similarity matching. Higher values require more precise matches."
    )

    # Button to run command
    if st.button("Run Command"):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Perform NER and process entities based on the selected input type
        if user_input:

            with st.spinner(f" 🚀 App Running"):
                with st.spinner(f" 🔄 Detecting Entities"):
                    user_input = '"' + user_input + '"'    
                   
                if input_type == "Person":
                        entities =  ner_model.perform_ner_with_sliding_window(user_input, label_suffixes, max_tokens=512, overlap=100)
                        entities, labels = concatenate_entities(entities)
                        entities = [re.sub(r"\[CLS\]|\[SEP\]|\[UNK\]", "", text).strip() for text in entities]
                        entities, labels = filter_entities_by_length(entities, labels, max_length=5)

                        if not one_name_person:
                            if spacy_filtering:
                                entities, labels = spacy_filter_entities(entities)
                                print('After Spacy entities', 'labels', entities, labels)
                                    
                        if entities is not None: 
                            st.markdown('<h2 style="text-align: left; color: black;">Person Entities:</h2>', unsafe_allow_html=True)
                            process_entities_by_type(entities, labels,"PER", similarity_threshold, profession=profession, city=None, one_name_person=one_name_person)
                        else:
                            st.warning(f"No {input_type.lower()} entities detecetd with the NER Model.")
                        
                    
                elif input_type == "Location":
                        user_input = remove_genitive_s(user_input)
                        entities =  ner_model.perform_ner_with_sliding_window(user_input, label_suffixes, max_tokens=512, overlap=100)
                        entities, labels = concatenate_entities(entities)
                        entities = [re.sub(r"\[CLS\]|\[SEP\]|\[UNK\]", "", text).strip() for text in entities]
                        entities = [replace_hyphen_whitespace(text) for text in entities]

                        if entities is not None: 
                            st.markdown('<h2 style="text-align: left; color: black;">Organization Entities:</h2>', unsafe_allow_html=True)
                            process_entities_by_type(entities,labels, "ORG", similarity_threshold, profession=None, city=city)
                        else:
                            st.warning(f"No {input_type.lower()} entities detecetd with the NER Model.")
                st.success("Search complete!")
        else:
            st.write("Please enter some text to run the command.")

    

if __name__ == "__main__":
    main()


#streamlit run app.py
