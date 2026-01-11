# Web Crawling: Lakes & Tallest Buildings Data Analysis

## Goal
Extract and analyze data from Wikipedia to:
1. Map the locations of lakes in Western Australia  
2. Identify and visualize the tallest buildings in the United States  

This project demonstrates web scraping, data cleaning, and visualization skills in Python.

---

## Tools Used
- Python (requests, BeautifulSoup, pandas, NumPy, matplotlib, seaborn)  
- Cartopy for geographic plotting  

---

## Approach
**Task 1: Lakes of Western Australia**
- Requested the Wikipedia page and parsed HTML tables  
- Cleaned and combined multiple tables into a single dataset  
- Split coordinates into numeric latitude and longitude  
- Plotted all lakes on a map of Australia using Cartopy  

**Task 2: Tallest Buildings in the United States**
- Requested the Wikipedia page and parsed HTML tables  
- Cleaned the height data and converted to numeric  
- Filtered buildings taller than 1000 feet  
- Created a bar chart showing the tallest buildings  

---

## Key Insights
- **Lakes dataset:** Able to map geographic locations of hundreds of lakes accurately  
- **Buildings dataset:** Identified all buildings over 1000 feet and visualized their heights  
- Demonstrates ability to handle real-world web data, clean it, and produce meaningful visualizations  
- Shows proficiency in Python, web scraping, data cleaning, and visualization  

---

## Project Files
- `web_crawling_lakes_buildings.py` → Main Python script  
- `Alexander_plot_lakes.png` → Map of lakes  
- `Alexander_plot_buildings.png` → Bar chart of tallest buildings
