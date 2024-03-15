from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.options import PageLoadStrategy
from tqdm import tqdm
import sys
import json


MAX_ITER = 100

def dump_raw_dataset(movie_id: str):
    reviews_jsonl = []
    url = f"https://www.imdb.com/title/tt{movie_id}/reviews/?ref_=tt_ov_rt"

    driver = webdriver.Chrome()
    driver.get(url)
    driver.implicitly_wait(2)
    review_error = 0
    page = 1
    while page < MAX_ITER:
        try:
            load_more = driver.find_element(By.ID, "load-more-trigger")
            driver.implicitly_wait(2)
            load_more.click()
        except:
            break

    reviews = driver.find_elements(By.CLASS_NAME, "review-container")
    for id,  review in tqdm(enumerate(reviews), total=len(reviews)):
        try:
            ftitle = review.find_element(By.CLASS_NAME, "title").text
            fcontent = review.find_element(By.CLASS_NAME, "content").get_attribute("textContent").strip()
            frating = review.find_element(By.CLASS_NAME, "rating-other-user-rating").text
            frating = int(frating.split("/")[0].strip())
            fsentiment = 0 if frating < 5 else 1
            fdate = review.find_element(By.CLASS_NAME, "review-date").text
            fname = review.find_element(By.CLASS_NAME, "display-name-link").text
            reviews_jsonl.append({
                "id": id,
                "title": ftitle,
                "content": fcontent,
                "rating": frating,
                "sentiment": fsentiment,
                "date": fdate,
                "name": fname
            })
        except:
            review_error+=1
            continue

    print(f"total reviews: {len(reviews_jsonl)}")
    print(f"review_error_when_parse:{review_error}")

    driver.quit()
    with open(f"data/raw/{movie_id}.jsonl", "w", encoding = "utf-8") as f:
        for review in reviews_jsonl:
            json.dump(review, f)
            f.write("\n")

if __name__ == "__main__":
    if len(sys.argv)< 2:
        print("Please provide movie id")
        exit(1)

    dump_raw_dataset(sys.argv[1])

