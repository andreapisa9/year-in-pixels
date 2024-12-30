import time
import string
from copy import deepcopy
import datetime
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch
from geopy.geocoders import Nominatim
from wordcloud import STOPWORDS

# Filter for this year's and last year's data
def check_year(year_to_check : datetime.date, reference : int):
    """
    Check if the year of a given datetime.date matches a reference year.

    Parameters
    ----------
    year_to_check : datetime.date
        The datetime.date object to check.
    reference : int
        The year to compare with.

    Returns
    -------
    bool
        True if the year of year_to_check matches the reference year, False otherwise.
    """
    return True if year_to_check.year == reference else False

# Function to generate a LinearSegmentedColormap
def generate_colormap_from_hex_list(palette_list : list[str]) -> LinearSegmentedColormap:
    """
    Generate a LinearSegmentedColormap from a list of hexadecimal color strings.

    Parameters
    ----------
    palette_list : list[str]
        List of hexadecimal color strings defining the color palette.

    Returns
    -------
    LinearSegmentedColormap
        A LinearSegmentedColormap object with the specified palette.
    """
    
    rgba_palette = [to_rgba('#'+color) for color in palette_list]
    cmap = LinearSegmentedColormap.from_list("custom_palette", rgba_palette, N=1024)

    return cmap

# Function to make rounded bars in bar plots
def round_bars(ax : plt.Axes):
    """
    Replace all patches in an axis with rounded corner versions of themselves.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to modify.

    Returns
    -------
    None
    """
    
    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        
        # Create a FancyBboxPatch with rounded corners
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=0,rounding_size=0.020",
            ec="none", fc=color,
            mutation_aspect=4
        )
        
        # Remove the old patch and add the new one
        patch.remove()
        new_patches.append(p_bbox)

    # Add all new patches to the axes
    for patch in new_patches:
        ax.add_patch(patch)

# Function to determine if a day should be marked as a sick day
def sick_day(row : pd.DataFrame) -> str:
    """
    Determine if a day should be marked as a sick day.

    Parameters
    ----------
    row : pd.DataFrame
        A row of a DataFrame containing the columns 'scores', 'Symptoms', and 'Medication'.

    Returns
    -------
    str
        A string containing either '💀' (skull emoji), '🤒' (sick emoji), or an empty string.
    """
    assert set(['scores', 'Symptoms', 'Medication']).issubset(set(row.index)), "Wrong columns."
    if isinstance(row['Symptoms'], float) or isinstance(row['Medication'], float) : return ''
    if len(row['Symptoms'] + row['Medication']) > 4.0:
        if row['scores'] < 2.0 : return '💀'
        else : return '🤒'
    return ''

# Function to determine if a day should be marked as a productive day
def productive_day(row : pd.DataFrame) -> str :
    """
    Determine if a day should be marked as a productive day.

    Parameters
    ----------
    row : pd.DataFrame
        A row of a DataFrame containing the column 'Productivity Rating'.

    Returns
    -------
    str
        A string containing either '💪' (flexed arm emoji), '😴' (sleeping emoji), or an empty string.
    """
    assert set(['Productivity Rating']).issubset(set(row.index)), "Wrong columns."
    if row['Productivity Rating'] >= 4.0 : return '💪'
    elif row['Productivity Rating'] == 0.0 : return '😴'
    else : return ''

# Function to check if an element is contained in a list or matches a single element
def in_(contained, container) -> bool:
    """
    Check if an element is contained in a list or matches a single element.

    Parameters
    ----------
    contained : object
        The element to check for containment.
    container : list or object
        A list of elements or a single element to check against.

    Returns
    -------
    bool
        True if the element is contained in the list or matches the single element, False otherwise.
    """
    
    if isinstance(container, list) : return contained in container
    else : return contained == container

# Function to collect all possible tag values
def tag_list(data : pd.DataFrame, tag : str):
    """
    Collect all possible tag values for a given column in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the column of interest.
    tag : str
        Name of the column for which to collect all possible tag values.

    Returns
    -------
    list[str]
        List of all possible tag values for the given column.
    """

    # collect all possible tag values
    masked_data = data[data[tag].notna()] # masked DataFrame
    tag_values = masked_data[tag].apply(lambda x : [x] if not isinstance(x, list) else x).sum() # list with duplicates
    tag_values = list(dict.fromkeys(tag_values)) # remove duplicates

    return tag_values

# Function to collect all possible terms
def term_list(document_corpus : pd.DataFrame, label : str) -> list[str]:
    """
    Collect all possible terms in a given column of a DataFrame.

    Parameters
    ----------
    document_corpus : pd.DataFrame
        DataFrame containing the column of interest.
    label : str
        Name of the column for which to collect all possible terms.

    Returns
    -------
    list[str]
        List of all possible terms for the given column.
    """

    document_corpus[label] = document_corpus[label].apply(lambda x : [y.strip(string.punctuation) for y in x.strip().lower().split()])
    terms = tag_list(document_corpus, label)
    terms.remove('')
    return terms

# Function to get coordinates
def get_coordinates(city : str):
    """
    Get the geographical coordinates (latitude and longitude) for a given city.

    This function uses the Nominatim geocoding service to find the coordinates
    of a specified city. Special cases for certain city names are handled to 
    ensure accurate results. If a location is found, the latitude and longitude
    are returned; otherwise, the function returns None for both values after 
    waiting for a short period.

    Parameters
    ----------
    city : str
        The name of the city for which to obtain coordinates.

    Returns
    -------
    tuple
        A tuple containing the latitude and longitude of the city, or (None, None)
        if the location could not be found.
    """
    
    # I went to Arenella (Sicily), not Arenella (Campania)

    if 'Arenella' in city:
       city = 'Arenella Syracuse'
       additional_arguments = dict(postalcode = '96100')
    
    else : additional_arguments = {}

    # Agrate stands for Agrate Brianza
    if 'Agrate' in city:
       city = 'Agrate Brianza'

    # Vibo stands for Vibo Valentia
    if 'Vibo' in city:
       city = 'Vibo Valentia'

    geolocator = Nominatim(user_agent="year-in-pixels")
    location = geolocator.geocode({'city' : city, **additional_arguments}, timeout=10)
    if location:
        return location.latitude, location.longitude
    else:
        time.sleep(1.1)
        return None, None
    
# Function to compute TF-IDF
def tf_idf(document_corpus : pd.DataFrame, label : str, stopwords : set = STOPWORDS) -> dict:
    """
    Compute TF-IDF for each term in a given column of a pandas DataFrame.

    This function takes a pandas DataFrame and a column name as input and
    computes the TF-IDF for each term in the specified column. The TF-IDF is
    then returned as a dictionary with the terms as keys and the computed
    TF-IDF values as values.

    Parameters
    ----------
    document_corpus : pd.DataFrame
        DataFrame containing the column for which to compute TF-IDF.
    label : str
        Name of the column for which to compute TF-IDF.
    stopwords : set, optional
        Set of stopwords to ignore when computing TF-IDF. Defaults to
        STOPWORDS.

    Returns
    -------
    dict
        Dictionary with terms as keys and computed TF-IDF values as values.
    """
    
    assert label in document_corpus.columns, f"{label} is not a valid column."
    
    # convert column dtype to string
    document_corpus[label] = document_corpus[label].apply(lambda x : str(x))

    # compute list of possible terms (avoiding STOPWORDS)
    terms = term_list(document_corpus, label)
    terms = list(set(terms) - stopwords)

    # compute term frequency and document frequency for each term
    tf = []
    df = {term : 0 for term in terms}
    for index, row in document_corpus.iterrows():
        previous_df = deepcopy(df)
        for term in row[label]:
            if term in df.keys() : df[term] += 1 
        tf_row = {term : df[term] - previous_df[term] for term in df}
        tf.append(tf_row)

    tf = pd.DataFrame(tf, dtype="Sparse[int]")

    # TF-IDF = tf(i,j) * log(N/df(i))
    tf_idf = tf.apply(lambda x : x * np.log(tf.shape[0] / df[x.name]), axis=0)

    return tf_idf.mean().to_dict()