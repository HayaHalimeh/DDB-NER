import os
os.environ['dir'] = "./" #"/Users/halimeh/Desktop/DDB-NER"
os.chdir(os.environ['dir'])


import streamlit as st
# Suppose repo_script.py is in the same directory as this Streamlit app,
# or adjust the import statement according to your directory structure.
import time
from german_ner.GermanNER import GermanNerModel
from functions.utils import *
from ddbcaller._ddbcaller_class import DDBCaller
from functions.callLobidGndApi import *
from functions.utils import *


print(os.getcwd())
os.environ['DDB_API_KEY']= read_access_token_from_file(token_file_path= "./ddb_access_token.txt")
ddb = DDBCaller(os.environ['DDB_API_KEY'])


model_name = "mschiesser/ner-bert-german"

label_suffixes = ["-PER",  "-ORG"]
ner_model  = GermanNerModel(model_name)
ddb_name = 'Deutsche Digitale Bibliothek'


from PIL import Image


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
            newData.append((255, 255, 255, 0))  # Make this pixel fully transparent
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
    st.markdown(" 🔍 Validate result by matching detected entity with the Deutsche Nationalbibliothek ")


    if isinstance(ddb_url, str) == True:
        query = ddb_url.split("/")[-1]
        url = f'https://d-nb.info/gnd/{query}/about'

        match = match_names(get_lodib_data(query), entity_name, ner_model)    # ddb_name, ddb_url,

        if match is not None and match == True:
            st.markdown(f""" 
            <li> Result verified from the Deutsche Nationalbibliothek: <a href="{url}" target="_blank">{url}</a></li>
            """, unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            st.warning('* No verified match from Deutsche Nationalbibliothek found')
            

# DDB Function
def process_entities_by_type(entities, label_type):

    """
    Process extracted entities of a specific type (PER or ORG), match them with the DDB, and perform similarity checks.
    """

    already_seen = list()
    ddb_url = None

    if label_type ==  "PER":
        input_type = "person"
    else:
        input_type = "location"
        
    types, values = concatenate_entities(entities)
   
    with st.spinner('App running...'):
        for entity, label in zip(types, values):
            
            if label != label_type:
                continue

            entity = normalize_text(entity)
            
            if entity in already_seen:
                continue

            already_seen.append(entity)
            st.success(f" * {input_type} entity: {entity} detected")
            
            with st.spinner(f" 🚀  Querying Deutsche Digitale Bibliothek for {input_type}: {entity} ..."):
           
                # Perform query based on entity type
                if label == "PER":
                    data = ddb.get_person(entity, {})
                elif label == "ORG":
                    data = ddb.get_organisation(entity, {})
                else:
                    data = ddb.get_query(entity, {})


                if data is None or data['numberOfResults'] == 0:
                    #sometimes the search in DDB in not tailored to the entity type, try geenral query
                    with st.spinner(f" 🔄 No match found for {input_type} {entity} in DDB. Trying general querying ..."):
                        data = ddb.get_query(entity, {})
                    
                if data is None or data['numberOfResults'] == 0:
                    st.write("* No match from the Deutsche Digitale Bibliothek found.")

                else:
                    st.write(f" ➡️ {data['numberOfResults']} possible matches found")
 
                    gnd_info = extract_gnd_info(data)
                    #st.write('gnd_info', gnd_info)

                    # Jaccard distance matching
                    with st.spinner(' 🚀  Running Jaccard Similarity Search ...'):
                        st.write('Jaccard Search')
                        
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
                        with st.spinner('🚀 Running Semantic Similarity Search ...'):
                            st.write('Cosine Search')
                            
                            
                            similarity_threshold = .9
                            ddb_url, max_similarity = find_cosine_best_match(ner_model, entity, gnd_info)
                            
                            if max_similarity >= similarity_threshold:

                                st.markdown(f""" <br> ✅ Semantic Search in DDB complete for {entity}
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                <li>Best match from DDB: <a href="{ddb_url}" target="_blank">{ddb_url}</a></li>
                                """, unsafe_allow_html=True)

                                st.markdown("</ul></div>", unsafe_allow_html=True)
                            else:
                                st.warning(" * No appropriate match found in DDB")

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
        st.warning(f"* No {input_type.lower()} entities found")





def main():

    def set_light_theme():
        config_path = "./config.toml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)  # Create .streamlit directory if not exists

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
    logo_path_ddb = "./logo_ddb.png"  # Path to DDB logo
    logo_path_sicp = "./sicp.png"  # Path to SICP logo
    logo_path_da = "./transparent_logo.png"  # Path to DA logo
    ddb_ner_path = "./ddb_ner_diagram.jpg"  # Path to your JPEG image

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
    - Extract entities from text.
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
    Welcome to the DDB NER App! This application allows you to extract named entities from text and find matches in the Deutsche Digitale Bibliothek and LOBID databases.
    
    **Instructions**:
    - Select the type of input: General, Person, or Location.
    - Enter the relevant text or name.
    - Click 'Run Command' to perform the analysis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)



    # Selection for input type
    input_type = st.radio("Select the type of input you want to provide:", ("General", "Person", "Location"))

    # Text input based on selected type
    user_input = ""

    
    if input_type == "Person":
        user_input = st.text_input("Enter the name of the person you want to search for:")
        user_input = user_input.title()
        user_input = f"{user_input} ist ein bekannter Künstler" if user_input else ""
    elif input_type == "Location":
        user_input = st.text_input("Enter the name of the location you want to search for:")
        user_input = user_input.title()
        user_input = f"Die {user_input} ist ein bekannter Ort" if user_input else ""
    elif input_type == "General":
        user_input = st.text_input("Enter your description here:")
        user_input = user_input.title()

    # Button to run command
    if st.button("Run Command"):

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Perform NER and process entities based on the selected input type
        if user_input:

            with st.spinner(f" 🚀 Running NER on the input text ..."):
                time.sleep(1)
                entities = ner_model.perform_ner(label_suffixes, '"' + user_input + '"')

            if entities is not None:    
                if input_type == "General":
                    st.markdown('<h2 style="text-align: left; color: black;">Detected Entities:</h2>', unsafe_allow_html=True)
                    process_entities_by_type(entities, "PER")
                    process_entities_by_type(entities, "ORG")
                elif input_type == "Person":
                    st.markdown('<h2 style="text-align: left; color: black;">Person Entities:</h2>', unsafe_allow_html=True)
                    process_entities_by_type(entities, "PER")
                elif input_type == "Location":
                    st.markdown('<h2 style="text-align: left; color: black;">Organization Entities:</h2>', unsafe_allow_html=True)
                    process_entities_by_type(entities, "ORG")
            else:
                st.warning(f"No {input_type.lower()} entities detecetd with the NER Model.")

            st.success("Search complete!")
        else:
            st.write("Please enter some text to run the command.")

    

if __name__ == "__main__":
    # Example usage
    #input_image_path = "./logo.png"
    #output_image_path = "./transparent_logo.png"
    #make_background_transparent(input_image_path, output_image_path, target_color=(255, 255, 255))

    main()
