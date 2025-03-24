import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import textwrap
import plotly.graph_objects as go
import dash
from dash import dcc, html


df_large = pd.read_csv("FTAD_Clean_2025_02_13 - FTAD.csv")
df_latlong = pd.read_csv("FTAD_Clean_2025_02_13 - lat_long.csv")

good_cols = ['ID', 'date', 'country_txt', 'provstate', 'city', 'attacktype1_txt', 'targtype', 'weaptype1_txt', 'weapsubtype1_txt', 'nkill', 'nwound']

df_condensed = df_large[good_cols]

df_latlong = df_latlong.rename(columns={'Location': 'city'})

df_combined = df_condensed.merge(df_latlong, on='city', how='left')

df_Israel = df_combined[df_combined['country_txt'] == 'Israel']

# Filter the df_Israel DataFrame for the selected areas
areas_of_interest = ['Central', 'Golan Heights', 'North Sinai', 'Northern', 'Southern', 'Tel Aviv', 'Jerusalem', 'Eilat', 'Haifa']
df_filtered = df_Israel[df_Israel['provstate'].isin(areas_of_interest)]

df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
##########
#Descriptive statistics
"""
df = df_filtered.copy()


# 1. General overview of the data
#print(df.info())

# 2. Descriptive statistics for numerical columns
numerical_stats = df[['nkill', 'nwound']].describe()
print("Numerical Descriptive Statistics:")
print(numerical_stats)

# 3. Count the number of attacks by type
attack_type_counts = df['attacktype1_txt'].value_counts()
print("\nNumber of Attacks by Type:")
print(attack_type_counts)

# 4. Count the number of attacks by target type
target_type_counts = df['targtype'].value_counts()
print("\nNumber of Attacks by Target Type:")
print(target_type_counts)

# 5. Count the number of attacks by weapon type
weapon_type_counts = df['weaptype1_txt'].value_counts()
print("\nNumber of Attacks by Weapon Type:")
print(weapon_type_counts)

# 6. Count the number of attacks by city
city_attack_counts = df['city'].value_counts().head(10)  # Top 10 cities with the most attacks
print("\nTop 10 Cities with the Most Attacks:")
print(city_attack_counts)

# 7. Group by year (if 'date' column is in datetime format)
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure date is in datetime format
df['year'] = df['date'].dt.year
yearly_attack_counts = df['year'].value_counts().sort_index()
print("\nNumber of Attacks per Year:")
print(yearly_attack_counts)

# 8. Calculate total deaths and injuries (if you want to sum kills and wounds)
total_kills = df['nkill'].sum()
total_wounds = df['nwound'].sum()
print(f"\nTotal Kills: {total_kills}")
print(f"Total Wounds: {total_wounds}")

############# 
# Killed and wounded
# 1. Filter the conditions

df = df_filtered.copy()

no_killed_no_wounded = df[(df['nkill'] == 0) & (df['nwound'] == 0)]
no_killed_but_wounded = df[(df['nkill'] == 0) & (df['nwound'] > 0)]
killed_and_wounded = df[(df['nkill'] > 0) & (df['nwound'] > 0)]

# 2. Calculate the number of attacks for each category
no_killed_no_wounded_count = no_killed_no_wounded.shape[0]
no_killed_but_wounded_count = no_killed_but_wounded.shape[0]
killed_and_wounded_count = killed_and_wounded.shape[0]

# 3. Calculate the total number of attacks
total_attacks = df.shape[0]

# 4. Calculate the percentage for each category
no_killed_no_wounded_percentage = (no_killed_no_wounded_count / total_attacks) * 100
no_killed_but_wounded_percentage = (no_killed_but_wounded_count / total_attacks) * 100
killed_and_wounded_percentage = (killed_and_wounded_count / total_attacks) * 100

# 5. Print the results
print(f"Number of attacks where no one was killed or wounded: {no_killed_no_wounded_count} ({no_killed_no_wounded_percentage:.2f}%)")
print(f"Number of attacks where no one was killed but people were wounded: {no_killed_but_wounded_count} ({no_killed_but_wounded_percentage:.2f}%)")
print(f"Number of attacks where people were both killed and wounded: {killed_and_wounded_count} ({killed_and_wounded_percentage:.2f}%)")

df = df_filtered.copy()

# 1. Filter the dataset to include only attacks where people were killed
attacks_with_kills = df[df['nkill'] > 0]

# 2. Calculate the average number of people killed
average_kills = attacks_with_kills['nkill'].mean()

# 3. Print the result
print(f"Average number of people killed in attacks where people were killed: {average_kills:.2f}")

##############
# deadliest weapons on average
df = df_filtered.copy()

# 1. Group by weapon type and calculate the average number of kills for each type
weapon_type_avg_kills = df.groupby('weaptype1_txt')['nkill'].mean().sort_values(ascending=False)

# 2. Group by weapon subtype and calculate the average number of kills for each subtype
weapon_subtype_avg_kills = df.groupby('weapsubtype1_txt')['nkill'].mean().sort_values(ascending=False)

# 3. Print the results with added explanation
print("Most deadly attacks on average by Weapon Type (average number of deaths when weapon type used):")
for weapon, avg_kills in weapon_type_avg_kills.head().items():
    print(f"{weapon}: {avg_kills:.2f} average deaths")

print("\nMost deadly attacks on average by Weapon Subtype (average number of deaths when weapon subtype used):")
for subtype, avg_kills in weapon_subtype_avg_kills.head().items():
    print(f"{subtype}: {avg_kills:.2f} average deaths")
"""
##########
# hot vs cold attacks
"""
print(df_filtered['weapsubtype1_txt'].unique())

df = df_filtered.copy()
df['year'] = df['date'].dt.year

cold = ['Knife or Other Sharp Object', 'Blunt Object', 'Rope or Other Strangling Device', 'Hands, Feet, Fists', 'Vehicle']
hot = ['Unknown Explosive Type', 'Automatic or Semi-Automatic Rifle','Letter Bomb', 'Grenade', 'Handgun', 'Time Fuse', 'Projectile (rockets, mortars, RPGs, etc.)', 'Rifle/Shotgun (non-automatic)', 'Unknown Gun Type',
       'Molotov Cocktail/Petrol Bomb', 'Other Explosive Type', 'Landmine', 'Arson/Fire', 'Suicide (carried bodily by human being)', 'Gasoline or Alcohol', 'Remote Trigger', 'Other Gun Type', 'Sticky Bomb']

# Filter the dataframe for hot and cold attacks
df['attack_type'] = df['weapsubtype1_txt'].apply(lambda x: 'cold' if x in cold else ('hot' if x in hot else 'other'))

# Group by year and attack type (hot/cold) to count the number of attacks
attack_counts = df[df['attack_type'].isin(['hot', 'cold'])].groupby(['year', 'attack_type']).size().unstack(fill_value=0)

# Create the line chart
fig = go.Figure()

# Add a line for cold attacks
fig.add_trace(go.Scatter(
    x=attack_counts.index, 
    y=attack_counts['cold'], 
    mode='lines+markers', 
    name='Cold Attacks', 
    line=dict(color='blue')
))

# Add a line for hot attacks
fig.add_trace(go.Scatter(
    x=attack_counts.index, 
    y=attack_counts['hot'], 
    mode='lines+markers', 
    name='Hot Attacks', 
    line=dict(color='red')
))

# Update layout for better visuals
fig.update_layout(
    title="Number of Hot and Cold Attacks Over Time",
    xaxis_title="Year",
    yaxis_title="Number of Attacks",
    template="plotly_dark",
    hovermode="x unified"
)

# Show the figure
fig.show()

fig.write_html("hot_vs_cold_attacks_over_time.html")
"""
#########
# Attack Types per Year : bar chart w slider
"""
df = df_filtered.copy()

df['year'] = df['date'].dt.year

# Initialize an empty list to store frames for each year
frames = []

# Generate a frame for each unique year
for year in df['year'].unique():
    # Filter data for the current year
    filtered_df = df[df['year'] == year]
    
    # Count the number of attacks by type
    attack_type_counts = filtered_df['attacktype1_txt'].value_counts().reset_index()
    attack_type_counts.columns = ['Attack Type', 'Number of Attacks']
    
    # Create a bar chart for this year
    bar_chart = go.Bar(
        x=attack_type_counts['Attack Type'],
        y=attack_type_counts['Number of Attacks'],
        name=str(year)
    )
    
    # Append the chart as a frame
    frames.append(go.Frame(
        data=[bar_chart],
        name=str(year)
    ))

# Create the layout with a slider
layout = go.Layout(
    title="Number of Attacks by Type",
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[dict(
            label="Play",
            method="animate",
            args=[None, dict(
                frame=dict(duration=500, redraw=True),
                fromcurrent=True
            )]
        )]
    )],
    sliders=[dict(
        active=0,
        currentvalue=dict(
            font=dict(size=20),
            visible=True,
            prefix="Year: ",
            xanchor="center"
        ),
        steps=[
            dict(
                args=[[str(year)], dict(frame=dict(duration=500, redraw=True), mode='immediate', transition=dict(duration=300))],
                label=str(year),
                method='animate'
            ) for year in df['year'].unique()
        ]
    )]
)

# Create the figure and add frames
fig = go.Figure(
    data=frames[0].data,  # Start with the data of the first year
    layout=layout,
    frames=frames
)

# Export the figure as an HTML file
fig.write_html("attack_types_by_year.html")

# Optionally, you can display the figure in Jupyter notebook (if using Jupyter)
#fig.show()
"""
############
# Weapons Type per Year : bar chart with slider
"""
df = df_filtered.copy()
df['year'] = df['date'].dt.year

# Initialize an empty list to store frames for each year
frames = []

# Generate a frame for each unique year
for year in df['year'].unique():
    # Filter data for the current year
    filtered_df = df[df['year'] == year]
    
    # Count the number of attacks by weapon type
    weapon_type_counts = filtered_df['weaptype1_txt'].value_counts().reset_index()
    weapon_type_counts.columns = ['Weapon Type', 'Number of Attacks']
    
    # Create a bar chart for this year
    bar_chart = go.Bar(
        x=weapon_type_counts['Weapon Type'],
        y=weapon_type_counts['Number of Attacks'],
        name=str(year)
    )
    
    # Append the chart as a frame
    frames.append(go.Frame(
        data=[bar_chart],
        name=str(year)
    ))

# Create the layout with a slider
layout = go.Layout(
    title="Number of Attacks by Weapon Type",
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[dict(
            label="Play",
            method="animate",
            args=[None, dict(
                frame=dict(duration=500, redraw=True),
                fromcurrent=True
            )]
        )]
    )],
    sliders=[dict(
        active=0,
        currentvalue=dict(
            font=dict(size=20),
            visible=True,
            prefix="Year: ",
            xanchor="center"
        ),
        steps=[
            dict(
                args=[[str(year)], dict(frame=dict(duration=500, redraw=True), mode='immediate', transition=dict(duration=300))],
                label=str(year),
                method='animate'
            ) for year in df['year'].unique()
        ]
    )]
)

# Create the figure and add frames
fig = go.Figure(
    data=frames[0].data,  # Start with the data of the first year
    layout=layout,
    frames=frames
)

# Export the figure as an HTML file
#fig.write_html("weapon_types_by_year.html")

# Optionally, you can display the figure in Jupyter notebook (if using Jupyter)
fig.show()
"""
###########
# Gun attacks over time
"""
df_filtered = df_filtered.copy()
df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')

guns = ['Unknown Gun Type', 'Automatic or Semi-Automatic Rifle', 'Handgun', 'Rifle/Shotgun (non-automatic)']
df_guns = df_filtered[df_filtered['weapsubtype1_txt'].isin(guns)]

# Extract year from the 'date' column and create a 'year' column
df_guns['year'] = df_guns['date'].dt.year

# Group by year and count the number of gun attacks for each year
df_yearly_counts = df_guns.groupby('year').size().reset_index(name='gun_attacks_count')

# Create a DataFrame with all years between 1972 and 2019
all_years = pd.DataFrame({'year': range(1973, 2020)})

# Merge with the gun attack counts, using 'left' join to include all years from 1973 to 2019
df_yearly_counts = pd.merge(all_years, df_yearly_counts, on='year', how='left')

# Fill any missing gun attack counts with 0 for years with no gun attacks
df_yearly_counts['gun_attacks_count'] = df_yearly_counts['gun_attacks_count'].fillna(0).astype(int)

# Create the plot with Plotly Express
fig = px.line(df_yearly_counts, x='year', y='gun_attacks_count', 
              title='Number of Gun Attacks per Year in Israel (1973-2019)', 
              labels={'year': 'Year', 'gun_attacks_count': 'Number of Gun Attacks'},
              markers=True)

# Set the x-axis to label every 5 years
fig.update_xaxes(tickmode='array', tickvals=list(range(1973, 2020, 2)))

# Show the plot
fig.show()

# Export the plot to an HTML file
fig.write_html("gun_attacks_per_year.html")
"""
##################
# Explosive attacks over time
"""
df_filtered = df_filtered.copy()
df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')

explosives = ['Explosives']
df_explosive = df_filtered[df_filtered['weaptype1_txt'].isin(explosives)]

# Extract year from the 'date' column and create a 'year' column
df_explosive['year'] = df_explosive['date'].dt.year

# Group by year and count the number of gun attacks for each year
df_yearly_counts = df_explosive.groupby('year').size().reset_index(name='explosive_attacks_count')

# Create a DataFrame with all years between 1972 and 2019
all_years = pd.DataFrame({'year': range(1973, 2020)})

# Merge with the gun attack counts, using 'left' join to include all years from 1973 to 2019
df_yearly_counts = pd.merge(all_years, df_yearly_counts, on='year', how='left')

# Fill any missing gun attack counts with 0 for years with no gun attacks
df_yearly_counts['explosive_attacks_count'] = df_yearly_counts['explosive_attacks_count'].fillna(0).astype(int)

# Create the plot with Plotly Express
fig = px.line(df_yearly_counts, x='year', y='explosive_attacks_count', 
              title='Number of Explosive Attacks per Year in Israel (1973-2019)', 
              labels={'year': 'Year', 'explosive_attacks_count': 'Number of Explosive Attacks'},
              markers=True)

# Set the x-axis to label every 5 years
fig.update_xaxes(tickmode='array', tickvals=list(range(1973, 2020, 2)))

# Show the plot
fig.show()

# Export the plot to an HTML file
fig.write_html("explosive_attacks_per_year.html")

"""
####################
# HEATMAP: WEAPON TYPE AND LOCATION (+ WEST BANK)
"""
# Filter the df_Israel DataFrame for the selected cities
areas_of_interest_wb = ['Central', 'Golan Heights', 'North Sinai', 'Northern', 'Southern', 'Tel Aviv', 'Jerusalem', 'Eilat', 'Haifa', 'West Bank']
df_filtered_wb = df_combined[df_combined['provstate'].isin(areas_of_interest_wb)]

# Use .loc to avoid the "SettingWithCopyWarning"
df_filtered_wb.loc[:, 'log_nkill'] = np.log1p(df_filtered_wb['nkill'])  # log1p to handle zero values


# Pivot the DataFrame to get 'city' as columns, 'weaptype1_txt' as rows, and 'nkill' as values
heatmap_data = df_filtered_wb.pivot_table(index='weaptype1_txt', columns='provstate', values='log_nkill', aggfunc='sum', fill_value=0)


heatmap_data.rename(index={'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)': 'Vehicle'}, inplace=True)


# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap=sns.color_palette("light:#5A9", as_cmap=True), fmt='.2f', linewidths=0.5)

# Add titles and labels
plt.title('Heatmap of Kill Count (normalized) by Area and Weapon Type')
plt.xlabel('Area')
plt.xticks(rotation=45)
plt.ylabel('Weapon Type')
plt.show()
"""

########
# HEATMAP: WEAPON AND LOCATION (ISRAEL ONLY)
"""
# Filter for selected areas
areas_of_interest_wb = ['Central', 'Golan Heights', 'North Sinai', 'Northern', 'Southern', 
                         'Tel Aviv', 'Jerusalem', 'Eilat', 'Haifa']
df_filtered_wb = df_combined[df_combined['provstate'].isin(areas_of_interest_wb)].copy()

# Apply log transformation to 'nkill' to normalize
df_filtered_wb['log_nkill'] = np.log1p(df_filtered_wb['nkill'])

# Pivot the DataFrame for heatmap (log-transformed data)
heatmap_data = df_filtered_wb.pivot_table(index='weaptype1_txt', columns='provstate', 
                                          values='log_nkill', aggfunc='sum', fill_value=0)

# Pivot the DataFrame for non-normalized 'nkill' (original data for hover)
heatmap_data_nkill = df_filtered_wb.pivot_table(index='weaptype1_txt', columns='provstate', 
                                                values='nkill', aggfunc='sum', fill_value=0)

# Rename the specific long label in 'weaptype1_txt' column
heatmap_data.rename(index={'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)': 'Vehicle'}, inplace=True)
heatmap_data_nkill.rename(index={'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)': 'Vehicle'}, inplace=True)

# Reset index for both heatmaps to make it easier to use Plotly
heatmap_data.reset_index(inplace=True)
heatmap_data_nkill.reset_index(inplace=True)

# Create a Plotly heatmap
fig = px.imshow(heatmap_data.set_index(['weaptype1_txt']).T,  # Transpose for Plotly's format
                labels=dict(x="Weapon Type", y="Area", color="Log Kill Count"),
                title='Heatmap of Kill Count (Normalized) by Area and Weapon Type')

# Create hover data as a dictionary (matching each cell with its non-normalized count)
hover_data = heatmap_data_nkill.set_index('weaptype1_txt').T.values

# Add hover data to the plot with custom hovertemplate
fig.update_traces(
    hovertemplate='Weapon Type: %{x}<br>Area: %{y}<br>Normalized Kill Count: %{z:.2f}<br>' +
                  'Non-Normalized Kill Count: %{customdata}<br>',
    customdata=hover_data
)

# Adjust layout for better visibility
fig.update_layout(
    width=1200,  # Increase the width
    height=800,  # Increase the height
    title_x=0.5,
    legend_title="Area",
    legend=dict(font=dict(size=14)),  # Increase the font size of the legend
    xaxis_title='Weapon Type',
    yaxis_title='Area',
    margin=dict(l=50, r=50, t=50, b=100),  # Add margin for better spacing
)

# Show the plot
fig.show()
fig.write_html("israel_weapon_location_heatmap.html")
"""
##########################
#WEAPON SUBTYPE HEATMAP
"""
# Filter for selected areas
areas_of_interest_wb = ['Central', 'Golan Heights', 'North Sinai', 'Northern', 'Southern', 
                         'Tel Aviv', 'Jerusalem', 'Eilat', 'Haifa']
df_filtered_wb = df_combined[df_combined['provstate'].isin(areas_of_interest_wb)].copy()

# Apply log transformation to 'nkill' to normalize
df_filtered_wb['log_nkill'] = np.log1p(df_filtered_wb['nkill'])

# Pivot the DataFrame for heatmap (log-transformed data)
heatmap_data = df_filtered_wb.pivot_table(index='weapsubtype1_txt', columns='provstate', 
                                          values='log_nkill', aggfunc='sum', fill_value=0)

# Pivot the DataFrame for non-normalized 'nkill' (original data for hover)
heatmap_data_nkill = df_filtered_wb.pivot_table(index='weapsubtype1_txt', columns='provstate', 
                                                values='nkill', aggfunc='sum', fill_value=0)

# Rename the specific long label in 'weaptype1_txt' column
#heatmap_data.rename(index={'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)': 'Vehicle'}, inplace=True)
#heatmap_data_nkill.rename(index={'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)': 'Vehicle'}, inplace=True)

# Reset index for both heatmaps to make it easier to use Plotly
heatmap_data.reset_index(inplace=True)
heatmap_data_nkill.reset_index(inplace=True)

# Create a Plotly heatmap
fig = px.imshow(heatmap_data.set_index(['weapsubtype1_txt']).T,  # Transpose for Plotly's format
                labels=dict(x="Weapon Type", y="Area", color="Log Kill Count"),
                title='Heatmap of Kill Count (Normalized) by Area and Weapon Type')

# Create hover data as a dictionary (matching each cell with its non-normalized count)
hover_data = heatmap_data_nkill.set_index('weapsubtype1_txt').T.values

# Add hover data to the plot with custom hovertemplate
fig.update_traces(
    hovertemplate='Weapon Type: %{x}<br>Area: %{y}<br>Normalized Kill Count: %{z:.2f}<br>' +
                  'Non-Normalized Kill Count: %{customdata}<br>',
    customdata=hover_data
)

# Adjust layout for better visibility
fig.update_layout(
    width=1200,  # Increase the width
    height=800,  # Increase the height
    title_x=0.5,
    legend_title="Area",
    legend=dict(font=dict(size=14)),  # Increase the font size of the legend
    xaxis_title='Weapon Type',
    yaxis_title='Area',
    margin=dict(l=50, r=50, t=50, b=100),  # Add margin for better spacing
)

# Show the plot
fig.show()
fig.write_html("israel_subweapon_location_heatmap.html")
"""
###########
## HEATMAP WITH SLIDER
"""

# Filter for selected areas
areas_of_interest_wb = ['Central', 'Golan Heights', 'North Sinai', 'Northern', 'Southern', 
                         'Tel Aviv', 'Jerusalem', 'Eilat', 'Haifa']
df_filtered = df_filtered.copy()  # Avoid SettingWithCopyWarning
df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')

# Filter for years 2005-2018
df_0518 = df_filtered[(df_filtered['date'].dt.year >= 2005) & (df_filtered['date'].dt.year <= 2018)].copy()

# Apply log transformation to 'nkill' to normalize
df_0518['log_nkill'] = np.log1p(df_0518['nkill'])

# Extract the 'year' from the 'date' column
df_0518['year'] = df_0518['date'].dt.year

# Ensure all weapon types are included
all_weapons = df_0518['weaptype1_txt'].unique()
df_0518['weaptype1_txt'] = df_0518['weaptype1_txt'].apply(lambda x: x if x in all_weapons else 'Other')

# Pivot the DataFrame for heatmap (log-transformed data)
heatmap_data = df_0518.pivot_table(index='weaptype1_txt', columns=['provstate', 'year'], 
                                   values='nkill', aggfunc='sum', fill_value=0)

# Create heatmap traces for each year
frames = []
for year in df_0518['year'].unique():
    # Filter data for this year
    df_year = heatmap_data.xs(year, level='year', axis=1)

    # Create a heatmap trace for this year
    trace = go.Heatmap(
        z=df_year.values,
        x=df_year.columns,
        y=df_year.index,
        colorscale='Viridis',
        colorbar=dict(title="Kill Count"),
        hovertemplate='Weapon Type: %{y}<br>Area: %{x}<br>Kill Count: %{z:.2f}<br>',
    )
    
    # Create a frame for the animation
    frames.append(go.Frame(data=[trace], name=str(year)))

# Create the figure
fig = go.Figure(
    data=[frames[0].data[0]],  # Set the initial frame
    layout=go.Layout(
        title="Heatmap of Kill Count (Normalized) by Area and Weapon Type",
        xaxis=dict(title="Weapon Type"),
        yaxis=dict(title="Area"),
        updatemenus=[dict(
            type="buttons",
            x=0.5,
            xanchor="center",
            y=-0.1,
            yanchor="top",
            buttons=[dict(
                args=[None, {"frame": {"duration": 300, "redraw": True},
                             "fromcurrent": True, "mode": "immediate",
                             "transition": {"duration": 300}}],
                label="Play",
                method="animate"
            )]
        )],
        sliders=[dict(
            steps=[dict(
                args=[[str(year)], {"frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate", "transition": {"duration": 300}}],
                label=str(year),
                method="animate"
            ) for year in df_0518['year'].unique()],
            currentvalue={'prefix': 'Year: ', 'visible': True, 'xanchor': 'center'}
        )],
        margin=dict(l=50, r=50, t=50, b=100),
    ),
    frames=frames
)

# Show the plot
fig.show()
"""

######
# CHART MONTHLY CASUALTIES
"""
# Ensure 'date' column is datetime
df_filtered = df_filtered.copy()  # Avoid SettingWithCopyWarning
df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')

# Filter for years 2005-2018
df_0518 = df_filtered[(df_filtered['date'].dt.year >= 2005) & (df_filtered['date'].dt.year <= 2018)].copy()

# Create 'month-year' column
df_0518['month_year'] = df_0518['date'].dt.to_period('M')

# Group by 'month-year' and 'provstate', summing 'nkill'
df_grouped = df_0518.groupby(['month_year', 'provstate'])['nkill'].sum().reset_index()

# Handle missing values in 'nkill' (if necessary)
df_grouped['nkill'] = df_grouped['nkill'].fillna(0)

# Convert 'month_year' back to string for Plotly
df_grouped['month_year'] = df_grouped['month_year'].astype(str)

# Sort by 'month_year' to ensure correct order in the plot
df_grouped = df_grouped.sort_values(by='month_year')

# Plot with Plotly for interactivity
fig = px.line(df_grouped, x='month_year', y='nkill', color='provstate', 
              title='Trends in Terrorist Attacks by Area (2005-2018)',
              labels={'nkill': 'Number of Kills', 'month_year': 'Month-Year'},
              template='plotly_dark')

# Customize hover effect
fig.update_traces(mode='lines+markers', marker=dict(size=4))  # Smaller dots

# Adjust chart size
fig.update_layout(width=900, height=500, legend_title_text='Province/State')

# Show the plot
fig.show()
fig.write_html("chart_monthly_casualties.html")
"""
#############################
"""
# Group by 'provstate' and 'weapsubtype1_txt', summing 'nkill'
df_bar = df_0518.groupby(['provstate', 'weapsubtype1_txt'])['nkill'].sum().reset_index()

# Handle NaN values in 'nkill' (if any)
df_bar['nkill'] = df_bar['nkill'].fillna(0)

# Create stacked bar chart
fig = px.bar(df_bar, x='provstate', y='nkill', color='weapsubtype1_txt',
             title='Kill Counts by Weapon Subtype Across Provinces/States',
             labels={'nkill': 'Number of Kills', 'provstate': 'Province/State', 'weapsubtype1_txt': 'Weapon Subtype'},
             barmode='stack',  # Stack bars
             template='plotly_dark')

# Adjust chart size
fig.update_layout(width=1000, height=600, legend_title_text='Weapon Subtype')

# Show plot
fig.show()
fig.write_html("stacked_bar_chart.html")

"""
"""

###############
# SCATTER MONTHLY ATTACKS

# Copy the dataframe to avoid modification of the original
df_filtered = df_filtered.copy()

# Convert 'date' to datetime format
df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')

# Filter for years 2005-2018
df_0518 = df_filtered[(df_filtered['date'].dt.year >= 2005) & (df_filtered['date'].dt.year <= 2018)].copy()

# Create 'month-year' column
df_0518['month_year'] = df_0518['date'].dt.to_period('M')

# Convert 'iyear' to integer to avoid any issues with float values
df_0518['iyear'] = df_0518['iyear'].astype(int)

# Convert 'weapsubtype1_txt' to string (to handle NaNs and floats)
df_0518['weapsubtype1_txt'] = df_0518['weapsubtype1_txt'].fillna('').astype(str)

# Aggregating data correctly by counting attacks and summing other relevant columns
agg_data = df_0518.groupby(['month_year', 'iyear', 'provstate']).agg(
    attacks=('ID', 'size'),  # Count the number of attacks (rows)
    total_killed=('nkill', 'sum'),
    total_wounded=('nwound', 'sum'),
    weapon_types=('weapsubtype1_txt', lambda x: ', '.join(x.unique()))  # Concatenate all weapon types for the group
).reset_index()

# Handle missing values for 'nkill' and 'nwound' (fill NaNs with 0)
agg_data['total_killed'].fillna(0, inplace=True)
agg_data['total_wounded'].fillna(0, inplace=True)

# Create the 'size' column for Plotly (based on the total number of people killed)
agg_data['size'] = agg_data['total_wounded'] + (agg_data['total_killed'] * 3) + 10  # Avoid zero size values

# Extract month and year for easier plotting
agg_data['month'] = agg_data['month_year'].dt.month
agg_data['year'] = agg_data['month_year'].dt.year

# Add a column for month names
agg_data['month_name'] = agg_data['month'].apply(lambda x: pd.to_datetime(f'2020-{x}-01').strftime('%B'))

# Keep only the necessary columns
agg_data = agg_data[['month', 'month_name', 'year', 'provstate', 'attacks', 'total_killed', 'total_wounded', 'weapon_types', 'size']]

# Create the scatter plot
fig = px.scatter(
    agg_data,
    x='month',  # Use month numbers (1 to 12) as the x-axis
    y='attacks',
    color='provstate',  # Color by area
    size='size',  # Size of dots based on number wounded
    hover_data={
        'month': False, 
        'year': True, 
        'attacks': True, 
        'total_killed': True, 
        'total_wounded': True, 
        'weapon_types': True,  # Show all weapon types in hover
        'size': False  # Exclude 'size' from hover
    },
    animation_frame='year',  # Slider for year
    title="Monthly Number of Attacks (2005-2018)",
    labels={"attacks": "Number of Attacks", "month": "Month", "year": "Year"}
)

# Update x-axis to show month names in correct order
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(1, 13)),  # Define months 1 to 12
        ticktext=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],  # Month names
    )
)

# Show the plot
fig.show()

fig.write_html("scatter_monthly_attacks.html")

"""
################
# SCATTER INDIVIDUAL ATTACKS

# Copy the dataframe to avoid modification of the original
df_filtered = df_filtered.copy()

# Convert 'date' to datetime format
df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')

# Filter for years 2005-2018
df_0518 = df_filtered[(df_filtered['date'].dt.year >= 2005) & (df_filtered['date'].dt.year <= 2018)].copy()

# Create 'month' and 'year' columns for easier plotting
df_0518['month'] = df_0518['date'].dt.month
df_0518['year'] = df_0518['date'].dt.year

# Handle missing values for 'nkill' and 'nwound' (fill NaNs with 0)
df_0518['nkill'].fillna(0, inplace=True)
df_0518['nwound'].fillna(0, inplace=True)

df_0518['size'] = (df_0518['nkill'] * 3) + 5  # Scaling factor to ensure visibility

# Create the scatter plot where each attack is a data point
fig = px.scatter(
    df_0518,
    x='month',
    y='nwound',  # Plot number of wounded on the y-axis
    color='provstate',  # Color by location area
    size='size',  # Size of the dot based on the number of people killed
    hover_data={
        'month': False,  # Hide month
        'year': True,  # Show year
        'nwound': True,  # Show number of wounded
        'nkill': True,  # Show number of killed
        'weapsubtype1_txt': True,  # Show weapon type used
        'size': False

    },
    animation_frame='year',  # Slider for year
    title="Individual Attacks (2005-2018)",
    labels={"nwound": "Number of Wounded", "month": "Month", "year": "Year"}
)

# Update x-axis to show month names
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(1, 13)),  # Define months 1 to 12
        ticktext=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],  # Month names
    )
)

# Prevent autoplay and start in a paused state
fig.update_layout(
    updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
)

fig.update_layout(
    title={
        'text': "Individual Attacks (2005-2018)",
        'x': 0.5,  # Centers the title
        'xanchor': 'center'
    },
    annotations=[
        dict(
            x=0.536,  # Centers the subtitle
            y=1.05,  # Adjusts position slightly above the title
            xref='paper',
            yref='paper',
            text="Each dot represents an individual attack. Dot size reflects number of deaths.",
            showarrow=False,
            font=dict(size=14)  # Adjust font size
        )
    ]
)

# Show the plot
fig.show()


fig.write_html('scatter_individual_attacks.html')
#print(df_filtered['provstate'].unique())

############
# MONTHLY INCIDENTS - JANUARY 2019
"""
# Creating the DataFrame
data_jan2019 = {
    "month": ["January"] * 13,
    "year": [2019] * 13,
    "nkill": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    "weapsubtype1_txt": [
        "Projectile (rockets, mortars, RPGs, etc.)",
        "Unknown Gun Type",
        "Pipe Bomb",
        "Unknown Gun Type",
        "Unknown Gun Type",
        "Pipe Bomb",
        "Knife or Other Sharp Object",
        "Knife or Other Sharp Object",
        "NaN",
        "Knife or Other Sharp Object",
        "Knife or Other Sharp Object",
        "Unknown Explosive Type",
        "Unknown Explosive Type"
    ]
}

df_jan2019 = pd.DataFrame(data_jan2019)

# Replacing string "NaN" with actual NaN values
df_jan2019.replace("NaN", pd.NA, inplace=True)
"""
"""
# Count the number of incidents per weapon type
weapon_counts = df_jan2019["weapsubtype1_txt"].value_counts()

# Plot bar chart
plt.figure(figsize=(10,5))
sns.barplot(x=weapon_counts.index, y=weapon_counts.values, palette="viridis")

plt.xticks(rotation=45, ha="right")
plt.xlabel("Weapon Type")
plt.ylabel("Number of Incidents")
plt.title("Number of Attacks by Weapon Type")
plt.show()
"""



