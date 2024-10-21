from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import time
import unicodedata
import webbrowser
import math
from pathlib import Path

import matplotlib.pyplot as plt
from langchain_community.document_loaders import PlaywrightURLLoader

try:
    import pyautogui
except Exception:
    pass
import pypdf
import requests
import tldextract

PDF_DOWNLOADED_PATH = Path("/home/leo/Documents/tmp_downloaded_pdfs/")


def clean_text(text):
    return ''.join(c for c in text if unicodedata.category(c) != 'Cs')


def save_json(obj, fp: Path | str):
    with open(fp, "w") as f:
        json.dump(obj, f)


def read_json(fp: Path | str):
    with open(fp, "r") as f:
        return json.load(f)


def write_txt(content: str, file_path: Path | str):
    try:
        with open(file_path, "w") as f:
            f.write(content)
    except UnicodeEncodeError:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except UnicodeEncodeError as e:
            print("UnicodeEncodeError")
            print(e)
            return False
    return True


def read_txt(file_path: Path | str):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError as e:
            print("UnicodeDecodeError")
            print(e)
            return None


def extract_txt_from_pdf(pdf_path) -> (int, list[str]):
    reader = pypdf.PdfReader(pdf_path)
    paper_content = []
    for i in range(len(reader.pages)):
        page_content = reader.pages[i].extract_text()
        paper_content.append(page_content)
    return len(reader.pages), paper_content


# Function to check if a PDF file is complete
def is_complete_pdf(file_path: Path):
    try:
        if Path(file_path).stat().st_size == 0:
            print(f"{file_path} is empty")
            return False
        with open(file_path, "rb") as fp:
            reader = pypdf.PdfReader(fp)
            # Try to get the number of pages to ensure it's a valid PDF
            try:
                num_pages = len(reader.pages)
            except Exception as e:
                print("trying decrypt with empty password...")
                reader.decrypt('')
                num_pages = len(reader.pages)
                print("trying decrypt with empty password... succeed!")
        return num_pages > 0
    except Exception as e:
        print(f"Error verifying PDF file {file_path}: {e}")
        return False


def download_pdf(pdf_url, pdf_path: str, timeout=10):
    try:
        paper_title = pdf_url.split("/")[-1]
        response = requests.get(pdf_url, stream=True, timeout=timeout)
        if response.status_code == 200:
            # pdf_path = save_dir / f"{paper_title}"
            with open(pdf_path, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        pdf_file.write(chunk)
            if is_complete_pdf(pdf_path):
                print(f"Downloaded and verified: {paper_title}")
                return str(pdf_path)
            else:
                print(f"Downloaded file {paper_title} is not a valid PDF")
                pdf_path.unlink()  # Delete the invalid PDF file
                return None
        else:
            print(f"Failed to download {paper_title} from {pdf_url}")
            return None
    except requests.exceptions.Timeout:
        print(f"Request timed out while downloading {pdf_url}")
        return None
    except Exception as e:
        print(f"Exception while downloading {pdf_url}: {e}")
        return None


def download_url_as_txt(url, save_path: str | Path, headless=True, timeout=10):
    loader = PlaywrightURLLoader(urls=[url], remove_selectors=["header", "footer"], headless=headless, timeout=timeout)
    data = loader.load()
    if len(data) == 0 or len(data[0].page_content) == 0:
        return None
    write_txt(data[0].page_content, save_path)
    return save_path


def download_pdf_webbrowser(pdf_url, save_path: str | Path, timeout=30, autoraise=True):
    if len(list(PDF_DOWNLOADED_PATH.iterdir())) != 0:
        for fp in PDF_DOWNLOADED_PATH.iterdir():
            os.remove(fp)
    assert len(list(PDF_DOWNLOADED_PATH.iterdir())) == 0, f"{PDF_DOWNLOADED_PATH} should be empty"

    webbrowser.open(pdf_url, autoraise=autoraise)
    start = time.time()
    print(f"webbrowser openning {pdf_url}")
    while True:
        if time.time() - start > timeout:
            print("time out!")
            pyautogui.hotkey("ctrl", "w")
            return None
        cur_files = list(PDF_DOWNLOADED_PATH.iterdir())
        if len(cur_files) == 1:
            downloaded_pdf = list(PDF_DOWNLOADED_PATH.iterdir())[0]
            if downloaded_pdf.name.endswith("crdownload") or downloaded_pdf.name.startswith(".com.google.Chrome"):
                print(f"{downloaded_pdf} still downloading")
                # change the timeout to wait longer for downloading the file
                timeout = 90
                time.sleep(2)
                continue

            if is_complete_pdf(downloaded_pdf):
                print(f"Downloaded and verified: {save_path}")
                # pdf_path = PDF_FOLDER / f"{paper_title}.pdf"
                shutil.move(downloaded_pdf, save_path)
                print(f"moved {downloaded_pdf} to {save_path}")
                return str(save_path)
            else:
                time.sleep(2)
                continue
        elif len(cur_files) > 1:
            print("Error: more than one file!")
            return None
        time.sleep(2)


def calculate_checksum(file_path, algorithm='md5'):
    """Calculate the checksum of a file using the specified algorithm.

    Args:
        file_path (str): Path to the file.
        algorithm (str): The hash algorithm to use (default is 'md5').

    Returns:
        str: The computed checksum in hexadecimal format.
    """
    # Create a hash object using the specified algorithm
    hash_obj = hashlib.new(algorithm)

    # Read the file in binary mode and update the hash object in chunks
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            hash_obj.update(chunk)

    # Return the computed checksum in hexadecimal format
    return hash_obj.hexdigest()


# def extract_domain(url):
#     try:
#         # Parse the URL
#         parsed_url = urlparse(url)
#
#         # Extract the domain name
#         domain = parsed_url.netloc
#
#         # Remove 'www.' prefix if present
#         if domain.startswith('www.'):
#             domain = domain[4:]
#
#         return domain
#     except Exception as e:
#         print(f"Error parsing URL {url}: {e}")
#         return None


def extract_domain(url):
    try:
        # Extract the domain components
        ext = tldextract.extract(url)

        # Combine the domain and suffix to get the main part of the domain
        main_domain = f"{ext.domain}.{ext.suffix}"

        return main_domain
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return None


def printValueCountsPercentage(df, name=None, denominator=None, topn=None):
    if name is None:
        cc = df.value_counts(dropna=False)
    else:
        cc = df[name].value_counts(dropna=False)
    if denominator is None:
        denominator = len(df)
    for idx, (n, v) in enumerate(dict(cc).items()):
        if topn is not None and idx >= topn:
            break
        printPercentage(v, denominator, n)


def printPercentage(value, total, prompt=""):
    print(f"{value} / {total} = {value / total * 100}% {prompt}")


def savefig(fig, save_dir: pathlib.Path, name, strip_title=False):
    if strip_title:
        p_title = fig.suptitle('').get_text()
        if len(fig.axes) == 1:
            a_title = fig.axes[0].get_title()
            fig.axes[0].set_title('')
    fig.savefig(
        (save_dir / '{}.pdf'.format(name)).as_posix(),
        bbox_inches='tight', dpi=500,
    )
    fig.savefig(
        (save_dir / '{}.png'.format(name)).as_posix(),
        bbox_inches='tight', dpi=500,
    )

    if strip_title:
        fig.suptitle(p_title)
        if len(fig.axes) == 1:
            fig.axes[0].set_title(a_title)


def randomize_df(df):
    return df.sample(frac=1).reset_index(drop=True)


# Function to calculate sample size
def calculate_sample_size(population_size, confidence_level=95, margin_of_error=0.05, population_proportion=0.5):
    # Z-scores for common confidence levels
    z_scores = {
        90: 1.645,
        95: 1.96,
        99: 2.576
    }

    # Get Z-score based on confidence level
    z = z_scores.get(confidence_level, None)
    if z is None:
        raise ValueError("Unsupported confidence level. Choose from 90, 95, or 99.")

    # Calculate initial sample size without FPC (for infinite population)
    p = population_proportion
    e = margin_of_error
    n = (z ** 2 * p * (1 - p)) / (e ** 2)

    # If population size is given, apply finite population correction (FPC)
    if population_size is not None:
        n_adj = n / (1 + (n - 1) / population_size)
        return math.ceil(n_adj)  # Return the adjusted sample size, rounded up
    else:
        return math.ceil(n)  # Return the sample size, rounded up for infinite population
