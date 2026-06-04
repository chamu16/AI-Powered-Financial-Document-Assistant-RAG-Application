import os
import re
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

# -----------------------------
# Bank websites to crawl
# -----------------------------

BANKS = {
    "commbank": {
        "domain": "commbank.com.au",
        "seeds": [
            "https://www.commbank.com.au/business",
            "https://www.commbank.com.au/business/loans-and-finance",
            "https://www.commbank.com.au/home-loans",
            "https://www.commbank.com.au/credit-cards",
            "https://www.commbank.com.au/banking",
            "https://www.commbank.com.au/insurance",
            "https://www.commbank.com.au/terms-and-conditions",
        ],
    },
    "anz": {
        "domain": "anz.com.au",
        "seeds": [
            "https://www.anz.com.au/personal/home-loans/",
            "https://www.anz.com.au/personal/credit-cards/",
            "https://www.anz.com.au/personal/bank-accounts/",
            "https://www.anz.com.au/personal/insurance/",
            "https://www.anz.com.au/business/loans-finance/",
            "https://www.anz.com.au/terms-conditions/",
        ],
    },
    "nab": {
        "domain": "nab.com.au",
        "seeds": [
            "https://www.nab.com.au/personal/home-loans",
            "https://www.nab.com.au/personal/credit-cards",
            "https://www.nab.com.au/personal/bank-accounts",
            "https://www.nab.com.au/personal/insurance",
            "https://www.nab.com.au/business/business-loans",
            "https://www.nab.com.au/common/terms-conditions",
        ],
    },
    "westpac": {
        "domain": "westpac.com.au",
        "seeds": [
            "https://www.westpac.com.au/personal-banking/home-loans/",
            "https://www.westpac.com.au/personal-banking/credit-cards/",
            "https://www.westpac.com.au/personal-banking/bank-accounts/",
            "https://www.westpac.com.au/personal-banking/insurance/",
            "https://www.westpac.com.au/business-banking/business-loans/",
            "https://www.westpac.com.au/about-westpac/terms-conditions/",
        ],
    },
}

# -----------------------------
# Crawl settings
# -----------------------------

MAX_PAGES_PER_BANK = 120
REQUEST_DELAY = 1.0
TIMEOUT = 20

OUTPUT_DIR = "data"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FinancialDocumentRAGBot/1.0; educational project)"
}

PDF_KEYWORDS = [
    "loan", "lending", "finance", "business", "home-loan", "home loan",
    "credit-card", "credit card", "card", "transaction", "savings",
    "account", "insurance", "pds", "product-disclosure",
    "terms", "conditions", "fees", "guide", "brochure",
    "policy", "privacy", "financial-services-guide", "fsg"
]

PAGE_KEYWORDS = [
    "loan", "lending", "finance", "business", "home-loan", "home",
    "credit-card", "credit", "cards", "bank-accounts", "accounts",
    "insurance", "terms", "conditions", "fees", "pds", "personal-banking",
    "business-banking"
]

CATEGORY_KEYWORDS = {
    "business_loans": ["business", "commercial", "sme", "business-loan", "business loan", "lending", "finance"],
    "home_loans": ["home-loan", "home loan", "mortgage", "housing"],
    "credit_cards": ["credit-card", "credit card", "card"],
    "transaction_accounts": ["transaction", "savings", "account", "bank-account", "bank account"],
    "insurance": ["insurance", "pds", "product-disclosure", "product disclosure"],
    "policies_terms": ["terms", "conditions", "fees", "guide", "policy", "privacy", "fsg"],
    "other_financial_docs": []
}

# -----------------------------
# Helper functions
# -----------------------------

def normalise_url(url):
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def is_same_domain(url, domain):
    netloc = urlparse(url).netloc.lower()
    return domain in netloc


def is_pdf_url(url):
    return ".pdf" in url.lower()


def is_relevant_pdf(url):
    lower_url = url.lower()
    return is_pdf_url(lower_url) and any(keyword in lower_url for keyword in PDF_KEYWORDS)


def is_relevant_page(url):
    lower_url = url.lower()
    return any(keyword in lower_url for keyword in PAGE_KEYWORDS)


def categorise_pdf(url):
    lower_url = url.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "other_financial_docs":
            continue

        if any(keyword in lower_url for keyword in keywords):
            return category

    return "other_financial_docs"


def safe_filename(url):
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)

    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"

    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)

    # Add short hash to avoid overwriting files with same name
    short_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    name = name.replace(".pdf", f"_{short_hash}.pdf")

    return name


def get_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            return None

        return response.text

    except Exception as e:
        print(f"[ERROR] Could not fetch page: {url} | {e}")
        return None


def extract_links(html, base_url, domain):
    soup = BeautifulSoup(html, "html.parser")

    page_links = set()
    pdf_links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        # Skip invalid frontend/template links
        if href.startswith("#"):
            continue

        if "{" in href or "}" in href:
            continue

        if "[" in href or "]" in href:
            continue

        if href.startswith("javascript:"):
            continue

        try:
            full_url = normalise_url(urljoin(base_url, href))
        except Exception:
            continue

        if not is_same_domain(full_url, domain):
            continue

        if is_pdf_url(full_url):
            if is_relevant_pdf(full_url):
                pdf_links.add(full_url)
        else:
            if is_relevant_page(full_url):
                page_links.add(full_url)

    return page_links, pdf_links


def download_pdf(pdf_url, bank_name):
    category = categorise_pdf(pdf_url)

    folder = os.path.join(OUTPUT_DIR, bank_name, category)
    os.makedirs(folder, exist_ok=True)

    filename = safe_filename(pdf_url)
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        print(f"[SKIP] Already exists: {filename}")
        return

    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            print(f"[SKIP] Not a PDF content type: {pdf_url}")
            return

        with open(path, "wb") as file:
            file.write(response.content)

        print(f"[DOWNLOADED] {bank_name} | {category} | {filename}")

    except Exception as e:
        print(f"[FAILED] {pdf_url} | {e}")


def crawl_bank(bank_name, config):
    domain = config["domain"]
    seeds = config["seeds"]

    visited_pages = set()
    discovered_pdfs = set()

    queue = deque(seeds)

    print(f"\n==============================")
    print(f"Starting crawl for: {bank_name.upper()}")
    print(f"==============================\n")

    while queue and len(visited_pages) < MAX_PAGES_PER_BANK:
        current_url = normalise_url(queue.popleft())

        if current_url in visited_pages:
            continue

        if not is_same_domain(current_url, domain):
            continue

        visited_pages.add(current_url)

        print(f"[CRAWLING] {current_url}")

        html = get_html(current_url)

        if html is None:
            continue

        page_links, pdf_links = extract_links(html, current_url, domain)

        for pdf in pdf_links:
            discovered_pdfs.add(pdf)

        for page in page_links:
            if page not in visited_pages:
                queue.append(page)

        time.sleep(REQUEST_DELAY)

    print(f"\nFound {len(discovered_pdfs)} relevant PDFs for {bank_name}.\n")

    for pdf_url in sorted(discovered_pdfs):
        download_pdf(pdf_url, bank_name)
        time.sleep(REQUEST_DELAY)

    print(f"\nFinished {bank_name}. Pages crawled: {len(visited_pages)} | PDFs found: {len(discovered_pdfs)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for bank_name, config in BANKS.items():
        crawl_bank(bank_name, config)

    print("\nAll downloads completed.")


if __name__ == "__main__":
    main()