from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

def search_url(search_term, page_number):
    template = 'https://www.myntra.com/{}?rawQuery={}&p={}'
    return template.format(search_term, search_term, page_number)

# Set up headless Chrome (optional: remove '--headless' if you want to see browser)
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
# chrome_options.add_argument("--headless")  # Uncomment for headless mode
chrome_options.add_argument("--disable-notifications")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Set up driver using WebDriver Manager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Get user input
org_url = input('Enter your search term: ').strip().replace(" ", "%20")

# Initialize lists
brands = []
price = []
original_price = []
description = []
product_url = []

# Scrape 10 pages
for i in range(1, 2):
    print(f"Scraping page {i}...")
    driver.get(search_url(org_url, i))
    time.sleep(2)  # Allow time for page to load

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Brand
    brand = soup.find_all('h3', class_="product-brand")
    for a in brand:
        brands.append(a.text.strip())

    # Discounted Price
    pr = soup.find_all('span', class_="product-discountedPrice")
    for b in pr:
        try:
            price.append(int(b.text.strip().replace('Rs. ', '').replace(',', '')))
        except:
            price.append(None)
    price.extend([None] * (len(brands) - len(price)))

    # Original Price
    mrp = soup.find_all('span', class_='product-strike')
    for c in mrp:
        try:
            original_price.append(int(c.text.strip().replace('Rs. ', '').replace(',', '')))
        except:
            original_price.append(None)
    original_price.extend([None] * (len(brands) - len(original_price)))

    # Description
    des = soup.find_all('h4', class_='product-product')
    for i in des:
        description.append(i.text.strip())
    description.extend([''] * (len(brands) - len(description)))

    # Product URLs
    li_elements = soup.find_all('li', class_="product-base")
    for d in li_elements:
        a_element = d.find('a', {'data-refreshpage': 'true', 'target': '_blank'})
        if a_element and 'href' in a_element.attrs:
            href = 'https://www.myntra.com' + a_element['href']
            product_url.append(href)
    product_url.extend([''] * (len(brands) - len(product_url)))

# Close the driver
driver.quit()

# Save data to DataFrame
df = pd.DataFrame({
    'Brand': brands,
    'Product Description': description,
    'Discounted Price (₹)': price,
    'Original Price (₹)': original_price,
    'Product URL': product_url
})

# Preview and export
print(df.head())
df.to_csv('myntra_products.csv', index=False)
print("✅ Data saved to 'myntra_products.csv'")
