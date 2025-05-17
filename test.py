import requests
from bs4 import BeautifulSoup
import os

# Step 1: Send an HTTP request to the webpage
url = 'https://example.com/reports'  # Replace with the actual URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
else:
    print('Failed to retrieve the webpage.')
    exit()

# Step 2: Find the document link
document_link = soup.find('a', {'class': 'download-link'})['href']

# Print the document link to verify
print('Document link found:', document_link)

# Handle relative URLs
base_url = 'https://example.com'  # The base URL of the website
full_url = os.path.join(base_url, document_link)

print('Full URL:', full_url)