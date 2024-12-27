import os
import sys
from pathlib import Path

CUR_DIR = Path(__file__).parent.resolve()
added_path = CUR_DIR.parent.resolve()
print(added_path)
sys.path.insert(0, str(added_path))

from src import helper
from src import common_path

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

if "GOOGLE_SEARCH_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_SEARCH_API_KEY not found in environment variables")
else:
    API_KEY = os.environ["GOOGLE_SEARCH_API_KEY"]

if "GOOGLE_SEARCH_CSE_ID" not in os.environ:
    raise EnvironmentError("GOOGLE_SEARCH_CSE_ID not found in environment variables")
else:
    CSE_ID = os.environ["GOOGLE_SEARCH_CSE_ID"]

GOOGLE_SEARCH_OUTPUT_DIR = common_path.OUT_PATH / "google_search_results"


def google_search(query, api_key, cse_id, num_results=10, start_index=1):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id,
        'num': num_results,
        'start': start_index,
    }
    response = requests.get(url, params=params)
    return response.json()


def generate_date_ranges(start_date, end_date, interval_days=1):
    current_start = start_date
    date_ranges = []

    while current_start < end_date:
        current_end = current_start + timedelta(days=interval_days)
        if current_end > end_date:
            current_end = end_date
        date_ranges.append((current_start, current_end))
        current_start = current_end

    return date_ranges


def get_total_results(query_res: dict):
    request_res = query_res["queries"]["request"][0]
    if "totalResults" in request_res:
        return int(request_res["totalResults"])
    return None


def has_next_page(query_res: dict):
    if "nextPage" in query_res["queries"]:
        return True
    else:
        return False


def query_pages(save_dir, num_pages, full_query, fp_prefix, results_per_page):
    total_res = None
    no_next_page = False
    for page in range(num_pages):
        print(f"querying: {full_query}")
        saved_fp = save_dir / f"{fp_prefix}_{page + 1}.json"
        if saved_fp.exists():
            prev_res = helper.read_json(saved_fp)
            total_res = get_total_results(prev_res)
            no_next_page = not has_next_page(prev_res)
            print(f"{saved_fp} already exists: {total_res=}, {no_next_page=}")
            continue

        start_index = page * results_per_page + 1
        if total_res is not None and start_index >= total_res:
            print(f"{saved_fp}: {start_index=} >= {total_res=}, skipping all the rest queries")
            break
        if no_next_page:
            print(f"{saved_fp}: no_next_page is True, skipping all the rest queries")
            break
        results = google_search(full_query, API_KEY, CSE_ID, num_results=results_per_page, start_index=start_index)

        # Save the results to a JSON file
        with open(saved_fp, 'w') as json_file:
            json.dump(results, json_file, indent=2)

        print(f'Saved: {saved_fp}')

        total_res = get_total_results(results)
        no_next_page = not has_next_page(results)
        print(f"after query: {total_res=}, {no_next_page=}")

        if "error" in results or "errors" in results:
            raise ValueError("error!")

        if no_next_page:
            print(f"after query: no_next_page is True, skipping all the rest queries")
            time.sleep(1)
            break
        time.sleep(1)


def search_and_save(query: str, save_dir, enable_query_date: bool = True):
    SAVE_DIR = Path(save_dir)
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    # Number of pages to retrieve
    num_pages = 10
    results_per_page = 10

    # Define start and end dates
    start_date = datetime(2023, 8, 10)
    end_date = datetime(2024, 8, 10)

    # if enable_query_date:
    #     # Generate date ranges
    #     date_ranges = generate_date_ranges(start_date, end_date, interval_days=1)
    #     # date_ranges.append((None, None))
    # else:
    #     date_ranges = [(None, None)]
    date_ranges = [(start_date, end_date)]

    print("date_ranges:", len(date_ranges))
    for dr in date_ranges:
        print(dr)

    for index, (start, end) in enumerate(date_ranges):
        # if start is None or end is None:
        #     full_query = query
        #     fp_prefix = "results_page_all_time"
        # else:
        date_query = f" after:{start.strftime('%Y-%m-%d')} before:{end.strftime('%Y-%m-%d')}"
        fp_prefix = f"results_page_one_year"
        full_query = query + date_query
        print(full_query)
        # exit(-1)
        query_pages(SAVE_DIR, num_pages, full_query, fp_prefix, results_per_page)


def search_and_save_addon(query: str, save_dir):
    SAVE_DIR = Path(save_dir)
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    # Number of pages to retrieve
    num_pages = 10
    results_per_page = 10
    # Define start and end dates
    start_date = datetime(2023, 8, 10)
    end_date = datetime(2024, 8, 10)

    # Check if `results_page_one_year_10.json` exists
    last_page_fp = SAVE_DIR / "results_page_one_year_10.json"
    if not last_page_fp.exists():
        return
    prev_res = helper.read_json(last_page_fp)
    total_results = int(prev_res["queries"]["previousPage"][0]["totalResults"])

    # If totalResults > 100, we need to split the date range
    if total_results > 100:
        # Calculate new interval_days based on the total number of results
        days_range = (end_date - start_date).days
        interval_days = max(1, days_range * 100 // total_results // 3)  # use one third of the required interval
        print(f"total_results: {total_results}, New interval_days: {interval_days}, ")
        date_ranges = generate_date_ranges(start_date, end_date, interval_days)
    else:
        return

    print("date_ranges:", len(date_ranges))
    for dr in date_ranges:
        print(dr)

    for index, (start, end) in enumerate(date_ranges):
        date_query = f" after:{start.strftime('%Y-%m-%d')} before:{end.strftime('%Y-%m-%d')}"
        fp_prefix = f"results_page_one_year_date_{index}"
        full_query = query + date_query
        print(f"querying {index}/{len(date_ranges)}:\n{full_query}")
        # exit(-1)
        query_pages(SAVE_DIR, num_pages, full_query, fp_prefix, results_per_page)


COMPANY_BLOGS = helper.read_json(common_path.DATA_PATH / "company_blogs.json")


def search_blog():
    query = (
        '"software" '
        'AND ("large language model" OR "foundation model" OR "large language models" OR "LLM" OR "LLMs" OR "FM" OR '
        '"foundation models" OR "FMs" OR "generative AI" OR "GenAI")'
        'site:{url}'
    )
    print(query.format(url=COMPANY_BLOGS["Apple"][0]))
    # exit(-1)
    for company, urls in COMPANY_BLOGS.items():
        print(f"===== searching for {company} =====")
        for url_idx, url in enumerate(urls):
            print(url)
            search_and_save(query.format(url=url), GOOGLE_SEARCH_OUTPUT_DIR / f"{company}/{url_idx}",
                            enable_query_date=False)


def main():
    search_blog()


if __name__ == '__main__':
    main()
