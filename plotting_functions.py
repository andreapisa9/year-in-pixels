from utils import *
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
from wordcloud import WordCloud
import plotly.express as px
import plotly.io as pio

# Function to generate a GitHub-style heatmap
def generate_pixels_heatmap(df : pd.DataFrame, palette : list[str], output_file : str, emoji : bool = False, font : str = 'Helvetica'):
    """
    Generates a GitHub-style heatmap chart.

    Parameters:
        df (pd.DataFrame): DataFrame with 'date' (datetime) and 'scores' columns.
        palette (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.
        emoji (bool, optional): If True, adds emojis to the heatmap cells based on the 'emoji' column in the DataFrame.
        font (str, optional): Font to use for emojis.

    Returns:
        None
    """

    assert 'emoji' in df.columns or not emoji, "Emoji condition not specified."

    # Ensure 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Create a full year of dates for the current year
    year = df['date'].dt.year.min()
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame for all days of the year
    all_days = pd.DataFrame({'date': all_dates})

    # Merge with the input DataFrame to align data
    df = all_days.merge(df, on='date', how='left').fillna({'rating': 0})
    data = df.copy()

    # Add week and day of the week for plotting
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.weekday

    # Handle the edge case where week 53 exists in the data
    if df['week'].max() == 53:
        df.loc[df['week'] == 53, 'week'] = 52

    # Ensure unique entries for pivot table by averaging ratings for duplicate days
    df = df.groupby(['week', 'day_of_week'], as_index=False)['scores'].mean()

    # Create a pivot table for heatmap data
    heatmap_data = df.pivot(index='day_of_week', columns='week', values='scores')

    # Create a custom colormap
    cmap = generate_colormap_from_hex_list(palette)

    # Plot the heatmap with rounded corners and spacing
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_aspect('equal')
    
    # Define the size of each square and the spacing
    square_size = 1.0
    spacing = 0.2
    prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc', size=10) # TODO : make portable to other OSs.

    # Draw each cell manually with rounded corners
    for y, (day, row) in enumerate(heatmap_data.iterrows()):
        for x, value in enumerate(row):
            if not np.isnan(value):
                color = cmap((value - 1.0) / 4.0)
                rect = FancyBboxPatch(
                    (x * (square_size + spacing), y * (square_size + spacing)),
                    square_size, square_size,
                    boxstyle=f"round,pad=0,rounding_size=0.2",
                    linewidth=0,
                    edgecolor=None,
                    facecolor=color
                )
                ax.add_patch(rect)
                that_day = data.iloc[y + x * 7, :]
                if emoji and that_day.emoji != '':
                    text = ax.annotate(that_day.emoji,
                                (x * (square_size + spacing) + 0.120, y * (square_size + spacing) + 0.9))
                    text.set(fontproperties=prop)

    # Add month labels at the top of the heatmap
    months = {'Jan' : 0,
              'Feb' : 4,
              'Mar' : 8,
              'Apr' : 13,
              'May' : 17,
              'Jun' : 21,
              'Jul' : 25,
              'Aug' : 30,
              'Sep' : 34,
              'Oct' : 38,
              'Nov' : 43,
              'Dec' : 47}
    for i, month in enumerate(months):
        ax.text(
            months[month] * (square_size + spacing),
            heatmap_data.shape[0] * (square_size + spacing) + 0.5,
            month,
            ha='left', va='bottom', color='white', fontsize=10, fontweight='bold', fontname=font
        )

    # Add weekday labels on the y-axis
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    first_weekday = start_date.day_of_week
    for i, weekday in enumerate(weekdays):
        y_pos = ((first_weekday + i) % 7) * (square_size + spacing)
        if ((first_weekday + i) % 7) % 2 == 1 : continue
        ax.text(
            -0.5,
            y_pos,
            weekday,
            ha='right', va='top', color='white', fontsize=10, fontweight='bold', fontname=font
        )

    # Adjust the limits and remove axes
    ax.set_xlim(0, heatmap_data.shape[1] * (square_size + spacing))
    ax.set_ylim(0, heatmap_data.shape[0] * (square_size + spacing))
    ax.invert_yaxis()
    ax.axis('off')

    # TODO: add possibility to rotate 90° clockwise

    # Save the heatmap to a SVG file with a transparent background
    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Function to plot a colorbar representing the ratings range and draw the average rating on it
def plot_average_rating(data : pd.DataFrame, palette_list : list[str], output_file : str, font : str = 'Helvetica'):
    """
    Plot a colorbar representing the ratings range and draw the average rating on it.

    Parameters:
        data (pd.DataFrame): DataFrame with 'scores' column.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.
        font (str, optional): Font to use for the average rating text.

    Returns:
        None
    """
    
    # Define the ratings range
    ratings_range = (0, 5)

    # Calculate the average rating
    average_rating = data['scores'].mean()

    # Create a LinearSegmentedColormap from the palette
    cmap = generate_colormap_from_hex_list(palette_list)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 1))

    range_x_axis = np.linspace(ratings_range[0], ratings_range[1], len(palette_list)*1024)

    # Draw the colorbar
    norm = Normalize(vmin=ratings_range[0], vmax=ratings_range[1])
    thinness = 20
    aspect = 1/thinness
    cb = ax.imshow(range_x_axis.reshape((1,len(palette_list)*1024)), cmap=cmap, aspect=aspect, norm=norm, extent=[ratings_range[0], ratings_range[1],0, thinness])
    ax.set_yticks([])
    ax.set_xticks([])

    # TODO: Draw the rating milestones
    # milestones = range_x_axis
    # for milestone in milestones:
    #     ax.scatter(milestone * 256 / ratings_range[1], 0.5, color=cmap(norm(milestone)), s=800, edgecolors='black', zorder=20)

    # Draw the average rating indicator
    ax.axvline(average_rating, color='white', linewidth=5, zorder=10)
    ax.text(ratings_range[0] - 0.1, 0.5, f'{average_rating:.2f}', color='white', ha='right', va='bottom', fontsize=48, fontweight='bold', fontname=font)
    fig.set_facecolor('black')

    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Line chart of avg rating per year
def plot_ratings_per_year(data_all : pd.DataFrame, label : str, palette_list : list[str], output_file : str, what : str = 'mean', font : str = 'Helvetica'):
    """
    Plot a line chart with value averages per year.

    Parameters:
        data_all (pd.DataFrame): DataFrame containing timed ratings.
        label (str): The column of ratings to be averaged.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.
        what (str, optional): The type of aggregation to perform. Can be 'mean' or 'sum'. Defaults to 'mean'.
        font (str, optional): Font to use for the axis labels. Defaults to 'Helvetica'.

    Returns:
        None
    """

    if label == 'emoji':
        data_all['emoji'] = data_all['emoji'].apply(lambda x : 1.0 if x != '' else 0.0)

    years = list(range(
        data_all.loc[0,['date']].to_numpy()[0].year,
        data_all.loc[data_all.shape[0]-1, ['date']].to_numpy()[0].year+1,
        1))

    avgs = []
    for year in years:
        filter = data_all['date'].apply(lambda x : check_year(x, year)) #TODO : check_year not needed anymore, substitute with data['date'].dt.year == year
        data_filtered = data_all[filter]

        if what == 'mean':
            avgs.append(data_filtered[label].mean())
        elif what == 'sum':
           avgs.append(data_filtered[label].sum()) 
        else:
            raise AttributeError(f'{what} is not an implemented aggregation.')

    fig, ax = plt.subplots(figsize=(5,4))

    for data_point in range(len(avgs)):
        ax.annotate(f'{avgs[data_point]:.2f}', (years[data_point]+0.1,avgs[data_point]), color='white', fontname=font, fontsize=15, fontweight='bold')
        if data_point > 0 :
            yoy_diff = avgs[data_point] - avgs[data_point - 1]
            sign = '' if yoy_diff < 0 else '+'
            color = '#'+palette_list[0] if yoy_diff < 0 else '#'+palette_list[-1]
            lateral_offset = years[data_point - 1] + 0.6
            vertical_offset = (avgs[data_point] + avgs[data_point - 1]) / 2.0
            ax.annotate(f'{sign}{yoy_diff:.2f}', (lateral_offset, vertical_offset), color=color, fontname=font, fontsize=12, fontweight='bold')
        ax.axvline(years[data_point], ymin=0.0, ymax=(avgs[data_point]-min(avgs))/(max(avgs)-min(avgs)) - 0.025, color='white', linestyle='--')

    ax.plot(years, avgs, 'o-', color='white', linewidth=3.5, markersize=10) #TODO : use sns.lineplot to obtain smoother curves
    ax.set_yticks([])
    ax.set_xticks(years, labels=years, fontname=font, fontweight='bold', fontsize=17, color='white')
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    ax.set_frame_on(False)

    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Rating frequency pie chart (with YoY increments if existing) viz function
def plot_rating_frequency_pie_chart(data_current_year : pd.DataFrame, palette_list : list[str], output_file : str, data_previous_year : pd.DataFrame = None, font : str = 'Helvetica'):
    """
    Plot a pie chart with rating frequencies of a given year, with Year over Year increments if existing.

    Parameters:
        data_current_year (pd.DataFrame): DataFrame containing timed ratings of the current year.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.
        data_previous_year (pd.DataFrame): DataFrame containing timed ratings of the previous year (optional).
        font (str): Font name to use for labels (optional, default='Helvetica').

    Returns:
        None.
    """

    # count rating frequencies
    rating_frequencies_current_year = data_current_year.groupby('scores').count()['date'].to_numpy(dtype=np.int32)

    # if YoY, compute previous year's measures
    if data_previous_year is not None:
        rating_frequencies_previous_year = data_previous_year.groupby('scores').count()['date'].to_numpy(dtype=np.int32)
        previous_year_pcts = rating_frequencies_previous_year / data_previous_year.shape[0] * 100.0

    # plot pie chart
    fig, ax = plt.subplots(figsize=(5,5))

    # adjust palette
    palette_list = ['#'+color for color in palette_list]

    # use emojis as labels
    prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc', size=23) #TODO : make portable to other OSs.

    # set properties of percentage text labels
    textprops = {
        'fontsize' : 10,
        'fontname' : font,
        'color' : 'white'
    }

    # draw pie chart
    _, texts, autotexts = ax.pie(x=rating_frequencies_current_year,
           labeldistance=0.90,
           labels=['😭', '😢', '😐', '🙂', '😁'],
           colors=palette_list,
           autopct=f"%.2f%%",
           textprops=textprops)
    
    for text in texts:
        text.set_fontproperties(prop)

    for t in range(len(autotexts)):
        if data_previous_year is None:
            autotexts[t].set_text(autotexts[t].get_text() + f" ({rating_frequencies_current_year[t]})")
        else:
            pct_difference = float(autotexts[t].get_text()[:-1]) - (previous_year_pcts[t])
            sign = '+' if pct_difference > 0 else ''
            autotexts[t].set_text(autotexts[t].get_text() + f" ({sign}{pct_difference:.2f})")

    fig.set_facecolor('black')
    ax.set_facecolor('black')

    ax.set_frame_on(False)

    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# (Normalized) Avg rating/productivity level per weekday/tag viz function, with YoY increments if existing
def plot_avg_ratings_per_weekday(data_all : pd.DataFrame, label : str, palette_list : list[str], output_file : str, font : str = 'Helvetica'):
    """
    Plot a bar chart with value averages per weekday.

    Parameters:
        data_all (pd.DataFrame): DataFrame containing timed ratings.
        label (str): The column of ratings to be averaged.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.
        font (str): Font name to use for labels (optional, default='Helvetica').

    Returns:
        None.
    """

    assert label in data_all.columns, f"No column named '{label}'"
    # assert data_all[label].dtype not a string and not a list

    curr_year = data_all.loc[data_all.shape[0]-1, 'date'].year

    total_avg = 0

    yearly_avgs = []
    for year in range (curr_year-1, curr_year+1):
        mask_year = data_all['date'].apply(lambda x : check_year(x, year))
        data_year = data_all[mask_year]
        data_year['weekday'] = data_year['date'].apply(lambda x : datetime.date.weekday(x))
        yearly_avgs.append(data_year.groupby('weekday', axis=0)[label].mean().to_numpy())
        if year == curr_year:
            total_avg = float(data_year[label].mean())

    fig, ax = plt.subplots(figsize=(9,4))

    yoy_diffs = [yearly_avgs[-1][i] - yearly_avgs[0][i] for i in range(7)]
    for j in range(7):
        yoy_diffs[j] = '+'+f'{yoy_diffs[j]:.2f}' if yoy_diffs[j] > 0 else f'{yoy_diffs[j]:.2f}'
        if '0.00' in yoy_diffs[j] : yoy_diffs[j] = '='

    container = ax.bar(range(7), yearly_avgs[-1], color='white', width=0.3)
    ax.bar_label(container, labels=[f"{yearly_avgs[-1][i]:.2f}({yoy_diffs[i]})" for i in range(7)], fontname=font, fontsize=11, color='white', padding=18)
    round_bars(ax)
    ax.axhline(total_avg, xmin=0.04, color='#'+palette_list[0], linestyle='--')
    ax.annotate('AVG', (-0.45, total_avg), fontname=font, fontsize=8, color='#'+palette_list[0], va='center')
    ax.set_yticks([])
    ax.set_xticks(range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontname=font, fontweight='bold', fontsize=17, color='white')
    ax.set_ylim(0, 5)
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    ax.set_frame_on(False)

    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Tag frequency viz function (descending sort by frequency)
def plot_tag_frequency(data_current_year : pd.DataFrame, tag_category : str, palette_list : list[str], output_file : str, font : str = 'Helvetica'):
    """
    Plot a horizontal bar chart with tags for a certain category, ordered by frequency.

    Parameters:
        data_current_year (pd.DataFrame): DataFrame containing one list of 'tag_category' tags per day.
        tag_category (str): The category of tags to count.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.
        font (str): Font name to use for labels (optional, default='Helvetica').

    Returns:
        None.
    """

    assert tag_category in data_current_year.columns, f"No tag category called {tag_category}"

    # collect all possible tag values
    tag_values = tag_list(data_current_year, tag_category)

    # count tag appearances
    tag_counters = {tag : 0 for tag in tag_values}
    for index, row in data_current_year[data_current_year[tag_category].notna()].iterrows(): # iterate over masked DataFrame to avoid null values
        tags_of_the_day = list(row[tag_category])
        for tag in tags_of_the_day:
            tag_counters[tag] += 1

    # sort for number of appearances (descending)
    tag_counters = pd.DataFrame(tag_counters.values(), index=tag_counters.keys(), columns=['count'])
    tag_counters.sort_values(by='count', ascending=True, inplace=True) # plotting function will flip the order anyways

    # draw horizontal bar chart
    fig, ax = plt.subplots(figsize=(9,4)) 

    data = tag_counters['count'].to_numpy(dtype=np.int32)

    container = ax.barh(y=range(0,len(tag_counters.index)), data=data, width=data, height=0.5, color='white')
    # round_bars(ax)
    ax.bar_label(container, data, color='white', fontname=font, fontsize=9, padding=6)

    ax.set_yticks(range(0,len(tag_counters.index)), labels=tag_counters.index, color='white', fontname=font, fontsize=11)
    ax.set_xticks([])
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    ax.set_frame_on(False)

    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Pairwise tag-tag correlation (Pearson) heatmap function
def plot_tag_to_tag_correlation_heatmap(data_current_year : pd.DataFrame, tag_1 : str, tag_2 : str, palette_list : list[str], output_file : str, font : str = 'Helvetica'):
    """
    Plot a 2D heatmap of pairwise Pearson correlations between tags in tag_1 and tag_2.

    Parameters:
        data_current_year (pd.DataFrame): DataFrame containing one list of 'tag_1' and 'tag_2' tags per day.
        tag_1 (str): The first category of tags to compute correlations for.
        tag_2 (str): The second category of tags to compute correlations for. If None, compute self-correlations for tag_1.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.

    Returns:
        None.
    """

    # compute tag-2-tag correlation
    # collect possible tag values
    tag_keys = tag_list(data_current_year, tag_1)
    tag_columns = tag_list(data_current_year, tag_2) if tag_2 is not None else tag_keys

    tag_keys = sorted(tag_keys) # ordered alphabetically (for now)
    tag_columns = sorted(tag_columns)

    df_columns = tag_keys + tag_columns

    one_hot_encoded_rows = []
    for index, row in data_current_year.iterrows():
        tag_1_values = list(row[tag_1]) if not isinstance(row[tag_1], float) else []
        if tag_2 is not None : tag_2_values = list(row[tag_2]) if not isinstance(row[tag_2], float) else []

        tag_1_one_hot_encoding = [1 if x in tag_1_values else 0 for x in tag_keys]
        if tag_2 is not None : tag_2_one_hot_encoding = [1 if y in tag_2_values else 0 for y in tag_columns]

        if tag_2 is not None : one_hot_encoded_rows.append(tag_1_one_hot_encoding + tag_2_one_hot_encoding)
        else : one_hot_encoded_rows.append([tag_1_one_hot_encoding])

    one_hot_encoded_rows = pd.DataFrame(one_hot_encoded_rows, index=data_current_year.index, columns=df_columns)

    tag_table = pd.concat([data_current_year['date'], one_hot_encoded_rows], axis=1)

    # TODO : order rows by number of appearances
    correlations = tag_table.corr(method='pearson')

    self_correlations = correlations.loc[tag_keys, tag_keys] # could need for some heatmap in the future?
    ethero_correlations = correlations.loc[tag_keys, tag_columns]

    correlations_to_plot = ethero_correlations if tag_2 is not None else self_correlations

    # Create a LinearSegmentedColormap from the palette
    cmap = generate_colormap_from_hex_list(palette_list)

    fig, ax = plt.subplots(figsize=(10,10))
    
    # draw 2d heatmap
    ax = sns.heatmap(correlations_to_plot, center=0.0, cmap=cmap, square=True, cbar=False)

    # TODO: (all notebook) make ticks invisible, but not labels
    cbar = ax.figure.colorbar(ax.collections[0], shrink=0.63)
    cbar.set_ticks(cbar.get_ticks()[1:-1], labels=cbar.get_ticks()[1:-1], fontname=font, color='white')
    ax.set_yticks(range(0,len(tag_keys)), labels=['\n'+key for key in tag_keys], fontname=font, color='white', va='top')
    ax.set_xticks(range(0,len(tag_columns)), labels=['\n'+col for col in tag_columns], fontname=font, color='white', ha='left')
    ax.set_facecolor('black')
    fig.set_facecolor('black')

    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# TODO: YoY?
# (Normalized) avg rating per tag viz function
def plot_avg_rating_per_tag(data_current_year : pd.DataFrame, label : str, palette_list : list[str], output_file : str, shrink : float = 0.0, font : str = 'Helvetica'):
    """
    Plot a horizontal bar chart with tags for a certain category, ordered by average rating of the tag.

    Parameters:
        data_current_year (pd.DataFrame): DataFrame containing one list of 'tag_category' tags per day.
        label (str): The category of tags to count.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting SVG image.
        shrink (float): Shrinkage parameter for computing the average rating (default=0.0).
        font (str): Font name to use for labels (optional, default='Helvetica').

    Returns:
        None.
    """
    
    assert label in data_current_year.columns, f"No column named '{label}'"
    # assert data_all[label].dtype not a string and not a list

    tag_values = tag_list(data_current_year, label)

    tag_values = sorted(tag_values)

    total_avg = 0

    yearly_avgs = []
    for tag in tag_values:
        data_current_year[str(tag)] = data_current_year[label].apply(lambda x : 1 if in_(tag, x) else 0)
        data_to_avg = data_current_year[data_current_year[str(tag)] == 1]
        yearly_avgs.append(data_to_avg['scores'].sum() / (data_to_avg.shape[0] + shrink)) # shrunk mean (penalize values with few appearances)
    
    total_avg = float(data_current_year['scores'].mean())

    fig, ax = plt.subplots(figsize=(9,4))

    # TODO: maybe for the future?
    # yoy_diffs = [yearly_avgs[-1][i] - yearly_avgs[0][i] for i in range(7)]
    # for j in range(7):
    #     yoy_diffs[j] = '+'+f'{yoy_diffs[j]:.2f}' if yoy_diffs[j] > 0 else f'{yoy_diffs[j]:.2f}'
    #     if '0.00' in yoy_diffs[j] : yoy_diffs[j] = '='

    # TODO: sort? Alphabetical for now

    container = ax.bar(range(len(tag_values)), yearly_avgs, color='white', width=0.3)
    ax.bar_label(container, labels=[f"{yearly_avgs[i]:.2f}" for i in range(len(tag_values))], fontname=font, fontsize=11, color='white', padding=18) #({yoy_diffs[i]})
    round_bars(ax)
    ax.axhline(total_avg, xmin=0.04, color='#'+palette_list[0], linestyle='--')
    ax.annotate('AVG', (len(tag_values)-0.65, total_avg+0.1), fontname=font, fontsize=8, color='#'+palette_list[0], va='center')
    ax.set_yticks([])

    rotation = 90 if len(tag_values) > 6 else 0 # I don't want productivity ratings rotated

    ax.set_xticks(range(len(tag_values)), labels=tag_values, fontname=font, fontweight='bold', fontsize=17, color='white', rotation=rotation)
    ax.set_ylim(0, 5)
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    ax.set_frame_on(False)

    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=1.0, transparent=True)
    plt.close()

# Map with visit frequency and average rating per location viz function
def plot_map(data_current_year : pd.DataFrame, palette_list : list[str], output_file : str, shrink : float = 0.1, font : str = 'Helvetica'):
    """
    Plot a map with visit frequency and average rating per location.

    Parameters:
        data_current_year (pd.DataFrame): DataFrame containing locations and scores for the current year.
        palette_list (list): List of hexadecimal color strings defining the color palette.
        output_file (str): Path to save the resulting map image.
        shrink (float): Shrinkage parameter for computing the average rating to avoid division by zero (default=0.1).
        font (str): Font name to use for annotations (optional, default='Helvetica').

    Returns:
        None.
    """
    
    # compute average rating per location
    tag_values = tag_list(data_current_year, 'Location') 

    tag_values = sorted(tag_values)

    avgs = []
    visits = []
    for tag in tag_values:
        data_current_year[str(tag)] = data_current_year['Location'].apply(lambda x : 1 if in_(tag, x) else 0)
        data_to_avg = data_current_year[data_current_year[str(tag)] == 1]
        visits.append(data_to_avg[str(tag)].sum())
        avgs.append(data_to_avg['scores'].sum() / (data_to_avg.shape[0] + shrink))

    location_data = pd.DataFrame({
        'Location' : {i : tag_values[i] for i in range(len(tag_values))},
        'color' : {i : avgs[i] for i in range(len(avgs))},
        'value' : {i : visits[i] for i in range(len(visits))}})
    
    # formatting to show the correct map details
    location_data = location_data.sort_values('value', ascending=False)
    location_data['text'] = [f"{city_name} ({color:.2f})" for city_name, color in location_data[['Location', 'color']].to_numpy()]

    # Apply the function to get coordinates
    location_data['latitude'], location_data['longitude'] = zip(*location_data['Location'].apply(get_coordinates))
    
    color_scale = ['#'+color for color in palette_list]

    fig = px.scatter_map(
        location_data,
        lat='latitude',
        lon='longitude',
        color='color',
        size='value',
        zoom=3.2,
        center={'lat':44.7, 'lon':13.5},
        text='text',
        map_style='carto-positron-nolabels',
        color_continuous_scale=color_scale,
        color_continuous_midpoint=3
    )

    fig.update_traces(
        mode='markers+text',
        textposition='top center',
        textfont={'size':15, 'color':'black'})
    
    fig.add_annotation(
        x=0,
        y=0,
        showarrow=False,
        text=f"Total locations visited: {location_data.shape[0]}",
        font={
            'size':18,
            'color':'black',
            'family':font
        }
    )

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # fig.show()

    pio.write_image(fig, output_file, width=800, height=600)

# Word cloud per daily notes (filtered by rating) viz function
def plot_wordcloud_from_daily_notes(data : pd.DataFrame, palette_list : list[str], output_file : str, ratings_range : tuple[int] = (0,6), img_path : str = './face-smile-solid.png', use_tf_idf : bool = True, stopwords : set = STOPWORDS):
    """
    Plot a word cloud from the daily notes in the given DataFrame, filtered to
    only include ratings in the given range. The word cloud is generated using
    either TF-IDF or simple term frequencies, depending on the use_tf_idf
    parameter. The resulting image is saved to the given output file.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing timed ratings.
    palette_list : list[str]
        List of hexadecimal color strings defining the color palette.
    output_file : str
        Path to save the resulting SVG image.
    ratings_range : tuple[int], optional
        Tuple of two integers specifying the range of ratings to include in
        the word cloud. The default is (0, 6).
    img_path : str, optional
        Path to the image used as a mask for the word cloud. The default is
        './face-smile-solid.png'.
    use_tf_idf : bool, optional
        Whether to use TF-IDF to compute word importance values. The default is
        True.
    stopwords : set, optional
        Set of words to ignore in the word cloud. The default is
        wordcloud.STOPWORDS.

    Returns
    -------
    None
    """

    # Filter data to include only the ratings specified in ratings_range
    assert len(ratings_range) == 2, "Ratings range is required to have length equal to 2."
    filtered_data = data[(data['scores'] > ratings_range[0]) & (data['scores'] < ratings_range[1])]
    
    mask = np.array(Image.open(img_path))

    if len(mask.shape) == 3 : # BW channels. Need only one
        mask = np.array([[j[1] for j in i] for i in mask])
    mask = np.abs(255-mask) # black -> white, white/transparent -> black
    
    cmap = generate_colormap_from_hex_list(palette_list)

    # Create wordcloud
    wc = WordCloud(mask=mask, colormap=cmap, background_color=None, mode='RGBA', stopwords=STOPWORDS, collocations=False)
    if tf_idf: # Compute TF-IDF word importance values
        term_importances = tf_idf(filtered_data, 'notes', stopwords)
        wc.generate_from_frequencies(term_importances)
    else: # generate from simple term frequencies
        text = filtered_data['notes'].sum() # Does this merge final/initial words of consecutive days?
        wc.generate(text)

    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    
    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=1.0, transparent=True)
    plt.close()