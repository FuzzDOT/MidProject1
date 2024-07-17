import os
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


class DataScraper:
    """
        A class for fetching data from webpages.
    """
    
    def __init__(self, url):
        """
            Constructs all the necessary attributes for the data scraper.
            
            Parameters
            ----------
                url : str
                    The address location of the website to fetch data from
        """
        self.url = url
        self.html_content = None
        self.soup = None
        self.text = None
        
    def fetch_content(self):
        """
            Fetch HTML content from URL.
        """
        try:
            res = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            res.raise_for_status() # Check if request was successful
            self.html_content = res.content
            print("Successfully fetched HTML content.")
        except Exception as e:
            print(f"Error fetching content: {e}")
            
    def parse_content(self):
        """
            Parses raw HTML content.
        """
        if self.html_content:
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
            print("Content parsed successfully.")
        else:
            print("No content to parse.")
            
    def retrieve_data(self):
        """
            Get data from parsed HTML content based on the tag and class name.
        """
        results = self.soup.find_all("div", class_="panel-panel-inner")
        description = results[1].find_all("div", class_="field-type-text-with-summary")
        cleantext = BeautifulSoup(description[0].text, "lxml").text.replace("\n", "").replace("\xa0", "")
        self.text = cleantext