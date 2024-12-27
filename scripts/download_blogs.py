import sys
from pathlib import Path

CUR_DIR = Path(__file__).parent.resolve()
added_path = CUR_DIR.parent.resolve()
print(added_path)
sys.path.insert(0, str(added_path))

from src import helper
from src import common_path

import pathlib
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import fitz  # PyMuPDF
from playwright.sync_api import sync_playwright

TRY_BROWSER = False

JSON_DIR = common_path.OUT_PATH / "google_search_results"
DOWNLOAD_DIR = Path(f"./{JSON_DIR.name}_txt")


def convert_pdf_to_txt(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # Write the extracted text to a file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return txt_path if txt_path.exists() else None


def download_url_as_pdf(url, pdf_path, headless=True):
    with sync_playwright() as p:
        # Launch the browser in headless or headed mode based on `headless` argument
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()

        # Open a new page in the browser
        page = context.new_page()

        # Navigate to the given URL
        page.goto(url)

        # Save the webpage as a PDF
        page.pdf(path=str(pdf_path), format="A4")

        # Close the browser
        browser.close()

    # Return the PDF path if it exists
    return pdf_path if pdf_path.exists() else None


def download_html_and_update(data_row, idx, lock, df, headless=False):
    # Check if the file has already been downloaded
    if not (pd.isna(data_row["downloaded"]) or data_row["downloaded"] == ""):
        print(f"{idx}: {data_row['link']} already downloaded")
        return None

    # Define paths for the PDF and the TXT file
    save_as_pdf = pathlib.Path(f"./tmp/{idx}.pdf")
    save_as_txt = DOWNLOAD_DIR / f"{idx}.txt"

    # Check if the TXT file already exists
    if save_as_txt.exists():
        downloaded_path = save_as_txt
    else:
        # Download the webpage as a PDF
        print(f"{idx}: Downloading {data_row['link']} as PDF")
        downloaded_pdf = download_url_as_pdf(data_row["link"], save_as_pdf, headless=headless)
        time.sleep(10)

        if downloaded_pdf:
            # Convert the downloaded PDF to a text file
            print(f"{idx}: Converting PDF to TXT")
            downloaded_txt = convert_pdf_to_txt(downloaded_pdf, save_as_txt)

            if downloaded_txt:
                print(f"{idx}: PDF successfully converted to TXT")
                downloaded_path = downloaded_txt
            else:
                print(f"{idx}: Failed to convert PDF to TXT")
                return None
        else:
            print(f"{idx}: Failed to download {data_row['link']} as PDF")
            return None

    # If successful, update the DataFrame with the path to the TXT file
    if downloaded_path:
        with lock:
            df.at[idx, "downloaded"] = str(downloaded_path)
    return idx


def download_htmls_from_csv(headless=True):
    global DOWNLOAD_DIR
    SAVED_CSV_FILE = JSON_DIR / "results_all.csv"
    df = pd.read_csv(JSON_DIR / "results_merged3.csv")
    print("before labelling:", len(df))

    print("after labelling:", len(df))
    if "downloaded" not in df.columns:
        df["downloaded"] = None
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    lock = threading.Lock()
    futures = []
    df["domain"] = df["link"].apply(helper.extract_domain)
    domain_done = []
    # for domain, gg in df.groupby("domain"):
    #     print(domain, len(gg))
    #     print(gg.values[0])
    if not headless:
        for idx, data_row in df.iterrows():
            assert idx == data_row["id"]
            # domain = helper.extract_domain(data_row["link"])
            # if domain in domain_done:
            #     print(domain, "already done")
            #     continue
            # domain_done.append(domain)
            download_html_and_update(data_row, idx, lock, df, headless)
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for idx, data_row in df.iterrows():
                assert idx == data_row["id"]
                if not (pd.isna(data_row["downloaded"]) or data_row["downloaded"] == ""):
                    print(f"{idx}: {data_row['link']} already downloaded")
                    continue
                futures.append(executor.submit(download_html_and_update, data_row, idx, lock, df, headless))

            for future in as_completed(futures):
                idx = future.result()
                if idx is not None:
                    print(f"{idx}: Completed")
            # time.sleep(5)
    df.to_csv(SAVED_CSV_FILE, index=False)


def gen_csv_all():
    file_pattern = "results_page_one_year_date_*.json"
    regex = re.compile(r'results_page_one_year_date_(\d+)_(\d+)\.json')
    df = {
        "title": [],
        "link": [],
        "company": [],
        "snippet": [],
        "url_idx": [],
        "date_idx": [],
        "page_idx": [],
    }
    saved_path = JSON_DIR / "results_all.csv"
    if saved_path.exists():
        raise ValueError(f"{saved_path} exists!")

    def extract_info(file):
        match = regex.match(file.name)
        if match:
            date_index, page_idx = match.groups()
            return int(date_index), int(page_idx)
        return None

    for company_name in JSON_DIR.iterdir():
        if not company_name.is_dir():
            continue
        for url_idx in company_name.iterdir():
            if not url_idx.is_dir():
                continue
            files = list(Path(url_idx).glob(file_pattern))
            # Step 3: Filter out files that do not match the pattern and extract info
            files_info = [(file, extract_info(file)) for file in files]
            files_info = [(file, page_idx) for file, page_idx in files_info if page_idx is not None]
            assert len(files) == len(files_info)
            # Step 4: Sort files by DATE1 and IDX
            # sorted_files = sorted(files_info, key=lambda x: x[1])
            sorted_files = sorted(files_info, key=lambda x: (x[1][0], x[1][1]))

            for (fp, (date_idx, page_idx)) in sorted_files:
                data = helper.read_json(fp)
                if "items" not in data:
                    print(fp, "no items found")
                    print(data)
                    continue
                for item in data["items"]:
                    df["title"].append(item["title"])
                    df["link"].append(item["link"])
                    df["snippet"].append(item["snippet"])
                    df["company"].append(company_name.name)
                    df["url_idx"].append(int(url_idx.name))
                    df["date_idx"].append(date_idx)
                    df["page_idx"].append(page_idx)

    df = pd.DataFrame.from_dict(df)
    print("before dropping duplication", len(df))
    df = df.drop_duplicates("link")
    print("after dropping duplication", len(df))
    df.to_csv(saved_path, index=False)
    print(df.groupby("company").apply(len))


def main():
    gen_csv_all()
    download_htmls_from_csv()
    download_htmls_from_csv(headless=False)


if __name__ == '__main__':
    main()
