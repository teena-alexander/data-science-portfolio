# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 13:01:56 2026

@author: 17323
"""

# Web Crawling Project: Olympics & Buildings Data Analysis

import requests

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ----------------------------
# Headers to mimic a real browser
# ----------------------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# ----------------------------
# Task 1: Mapping Lakes of Western Australia
# ----------------------------
url_lakes = "https://en.wikipedia.org/wiki/List_of_lakes_of_Western_Australia,_A%E2%80%93C"

# Request the page
page = requests.get(url_lakes, headers=headers)
print("Lakes page status:", page.status_code)

# Read all tables
tables = pd.read_html(page.text, encoding='utf-8')

# Combine all tables into one DataFrame
lakes_df = pd.concat(tables, ignore_index=True)

# Keep only Name and Coordinates
lakes_df = lakes_df[['Name', 'Coordinates']]

# Remove unwanted characters
lakes_df['Coordinates'] = lakes_df['Coordinates'].str.replace('\ufeff', '')

# Keep rows with '/' separating lat and lon
lakes_df = lakes_df[lakes_df['Coordinates'].str.contains('/')].copy()

# Split latitude and longitude
temp = lakes_df['Coordinates'].map(lambda s: s.split('/')[1])
lakes_df[['Latitude', 'Longitude']] = temp.str.split(expand=True)

# Convert to numeric values
lakes_df['Latitude'] = pd.to_numeric(
    lakes_df['Latitude'].map(lambda s: s.replace('°N', '') if 'N' in s else "-" + s.replace('°S', ''))
)
lakes_df['Longitude'] = pd.to_numeric(
    lakes_df['Longitude'].map(lambda s: s.replace('°E', '') if 'E' in s else "-" + s.replace('°W', ''))
)

# Plot lakes on map
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.scatter(lakes_df['Longitude'], lakes_df['Latitude'], color='red', marker='o', transform=ccrs.PlateCarree())
plt.title('Locations of Lakes in Western Australia')
plt.savefig('Alexander_plot_lakes.png', dpi=300)
# plt.show()

# ----------------------------
# Task 2: Tallest Buildings in the United States
# ----------------------------
url_buildings = 'https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_the_United_States'

# Request the page
response = requests.get(url_buildings, headers=headers)
print("Buildings page status:", response.status_code)

# Read tables
tables_buildings = pd.read_html(response.text, encoding='utf-8')

# First table contains the building data
buildings_df = tables_buildings[0]

# Keep relevant columns
buildings_df = buildings_df[['Name', 'Height ft (m)', 'Floors', 'Year']]

# Clean height column
buildings_df['Height'] = buildings_df['Height ft (m)'].str.split('f').str[0].str.replace(',', '').str.strip()
buildings_df['Height'] = buildings_df['Height'].astype('int64')

# Filter buildings at least 1000 feet tall
buildings_1000_df = buildings_df[buildings_df['Height'] >= 1000]

# Plot buildings
plt.figure(figsize=(14, 10))
sns.barplot(data=buildings_1000_df, y="Height", x="Name")
plt.xticks(rotation=90)
plt.title('Buildings in US at least 1000 feet tall')
plt.grid(True)
plt.tight_layout()
plt.savefig('Alexander_plot_buildings.png', dpi=300)
# plt.show()
