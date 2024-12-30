from plotting_functions import *
import os
import json
import pandas as pd
import matplotlib
from wordcloud import STOPWORDS
from dotenv import dotenv_values

if __name__ == "__main__":
    if not os.path.exists('./media'):
        os.makedirs('./media')

    # load externally defined environment variables
    env_vars = dotenv_values()

    # load color palette from environment variables
    palette_list = [env_vars["COLOR_PALETTE_{}".format(i)] for i in range(1,6)]
    palette_list

    matplotlib.use(f"module://mplcairo.{env_vars['CAIRO_VERSION']}") # Change to '.tk' if you have Linux or Windows

    # data extraction into Pandas DataFrame
    year = env_vars['YEAR']

    with open('./data/data_{}.json'.format(year)) as fp:
        data = json.load(fp)
    data = pd.DataFrame(data)
    data = data.drop(columns=['type'])

    # data cleaning
    data['scores'] = data['scores'].apply(lambda x : int(x[0])) # convert scores from lists to integers
    data['date'] = data['date'] = pd.to_datetime(data['date']) # convert dates from strings to datetime objects

    # make a column for each tag
    tags = {}
    for index, row in data.iterrows():
        row_data = {}
        tags_list = row['tags']
        for tag in tags_list:
            if tag['type'] not in row_data:
                row_data = {**row_data, tag['type'] : tag['entries']}
            else:
                row_data[tag['type']].extend(tag['entries'])
        tags = {**tags, index : row_data}

    tags = pd.DataFrame(tags).T

    data = pd.concat([data, tags], axis=1).drop(columns=['tags'])

    data['Productivity Rating'] = data['Productivity Rating'].apply(lambda x : int(x[0]) if isinstance(x, list) else 0) # convert productivity ratings from list to int

    data_all = data

    mask_previous_year = data_all['date'].apply(lambda x : check_year(x, int(year)-1))
    data_previous_year = data_all[mask_previous_year]

    mask_current_year = data_all['date'].apply(lambda x : check_year(x, int(year)))
    data_current_year = data_all[mask_current_year]

    # Font setting (change at will)
    font = env_vars['FONT']

    # Number of symptoms + number of pharmaceuticals calendar viz function, ratings in semi-transparency, skull 💀 emoji if bad day + sick day
    data_sick_days = data_all.copy()
    # data_sick_days
    data_sick_days['emoji'] = data_sick_days.apply(sick_day, axis=1)
    data_sick_days_current_year = data_sick_days[data_sick_days['date'].dt.year == 2024]

    data_productive_days = data_current_year.copy()

    data_productive_days['emoji'] = data_productive_days.apply(productive_day, axis=1)

    additional_stopwords = [
    "i'm",
    "it's",
    "kinda",
    "said",
    "overall",
    "anyways"
    ]

    stopwords = set(STOPWORDS)

    for word in additional_stopwords:
        stopwords.add(word)

    # Plotting functions
    generate_pixels_heatmap(data_current_year.copy(), palette_list, 'media/pixels_grid.svg', font=font)
    generate_pixels_heatmap(data_sick_days_current_year, palette_list, 'media/sick_days_heatmap.svg', True, font=font)
    generate_pixels_heatmap(data_productive_days, palette_list, 'media/productive_days_heatmap.svg', emoji=True, font=font)
    plot_average_rating(data_current_year.copy(), palette_list, 'media/avg_rating_curr_year.svg', font=font)
    plot_ratings_per_year(data_all.copy(), 'scores', palette_list, 'media/avg_ratings_per_year.svg', font=font)
    plot_ratings_per_year(data_sick_days, 'emoji', palette_list, 'media/sick_days_per_year.svg', what='sum', font=font)
    plot_ratings_per_year(data_all.copy(), 'Productivity Rating', palette_list, 'media/avg_productivity_per_year.svg', font=font)
    plot_rating_frequency_pie_chart(data_current_year.copy(), palette_list, 'media/rating_frequency_pie_chart.svg', data_previous_year, font=font)
    plot_avg_ratings_per_weekday(data_all.copy(), 'scores', palette_list, 'media/avg_ratings_per_weekday.svg', font=font)
    plot_avg_ratings_per_weekday(data_all.copy(), 'Productivity Rating', palette_list, 'media/avg_productivity_per_weekday.svg', font=font)
    plot_tag_frequency(data_current_year.copy(), 'Emotions', palette_list, 'media/emotion_frequency.svg', font=font)
    plot_tag_frequency(data_current_year.copy(), 'Activities', palette_list, 'media/activity_frequency.svg', font=font)
    plot_tag_to_tag_correlation_heatmap(data_current_year.copy(), 'Activities', 'Emotions', palette_list, 'media/activity_emotion_correlation.svg', font=font)
    plot_tag_to_tag_correlation_heatmap(data_current_year.copy(), 'Location', 'Emotions', palette_list, 'media/location_emotion_correlation.svg', font=font)
    plot_avg_rating_per_tag(data_current_year.copy(), 'Productivity Rating', palette_list, 'media/avg_rating_per_productivity.svg', font=font)
    plot_avg_rating_per_tag(data_current_year.copy(), 'Location', palette_list, 'media/avg_rating_per_location.svg', shrink=0.666, font=font)
    plot_avg_rating_per_tag(data_current_year.copy(), 'Emotions', palette_list, 'media/avg_rating_per_emotion.svg', shrink=0.666, font=font)
    plot_avg_rating_per_tag(data_current_year.copy(), 'Activities', palette_list, 'media/avg_rating_per_activity.svg', shrink=0.666, font=font)
    plot_avg_rating_per_tag(data_current_year.copy(), 'Medication', palette_list, 'media/avg_rating_per_medication.svg', font=font)
    plot_avg_rating_per_tag(data_current_year.copy(), 'Symptoms', palette_list, 'media/avg_rating_per_symptom.svg', shrink=0.666, font=font)
    plot_map(data_current_year.copy(), palette_list, 'media/map.svg', shrink=0.2, font=font)
    plot_wordcloud_from_daily_notes(data_current_year.copy(), palette_list, 'media/wordcloud_all_days.svg', stopwords=stopwords)
    plot_wordcloud_from_daily_notes(data_current_year.copy(), palette_list, 'media/wordcloud_bad_days.svg', (0,3), img_path='./face-sad-cry-solid.png', stopwords=stopwords)
    plot_wordcloud_from_daily_notes(data_current_year.copy(), palette_list, 'media/wordcloud_good_days.svg', (4,6), stopwords=stopwords)