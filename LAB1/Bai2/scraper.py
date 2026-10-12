import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://books.toscrape.com/"
books = []

while url:
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    for book in soup.select("article.product_pod"):
        link = book.select_one("h3 a")

        books.append({
            "title": link["title"],
            "price": book.select_one(".price_color").text,
            "availability": book.select_one(".availability").get_text(strip=True),
            "rating": book.select_one(".star-rating")["class"][1],
        })

    next_page = soup.select_one("li.next a")
    if next_page:
        url = urljoin(url, next_page["href"])  
    else: 
        url = None

df = pd.DataFrame(books)
df.to_csv("books.csv", index=False, encoding="utf-8-sig")

print(f"Collected {len(df)} books")