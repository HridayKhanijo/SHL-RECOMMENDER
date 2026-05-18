import json
import logging
import time
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/html,*/*",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}

def scrape_all():
    assessments = []
    seen_urls = set()

    # SHL catalog paginates with start= parameter, type=1 and type=2
    for type_id in [1, 2]:
        start = 0
        while True:
            url = f"https://www.shl.com/solutions/products/product-catalog/?start={start}&type={type_id}"
            logger.info(f"Fetching: {url}")
            try:
                time.sleep(1.5)
                resp = requests.get(url, headers=HEADERS, timeout=20)
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"Failed: {e}")
                break

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find all product rows in the catalog table
            rows = soup.select("tr.catalogue__row, [data-course-id], .catalogue__row")
            
            if not rows:
                # Try alternative selectors
                rows = soup.select("table tr")
                rows = [r for r in rows if r.select_one("td")]

            if not rows:
                logger.info(f"No rows found at start={start}, type={type_id} — stopping.")
                break

            found_new = False
            for row in rows:
                try:
                    # Get all links in this row
                    link = row.select_one("a[href*='product-catalog']")
                    if not link:
                        link = row.select_one("td a")
                    if not link:
                        continue

                    name = link.get_text(strip=True)
                    href = link.get("href", "")
                    if not href or not name:
                        continue

                    full_url = href if href.startswith("http") else f"https://www.shl.com{href}"

                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)
                    found_new = True

                    # Get test type badges
                    cells = row.select("td")
                    type_codes = []
                    for cell in cells:
                        spans = cell.select("span")
                        for span in spans:
                            text = span.get_text(strip=True)
                            if text in TEST_TYPE_MAP:
                                type_codes.append(text)

                    primary_type = type_codes[0] if type_codes else "K"

                    # Get description from cells
                    desc = ""
                    if len(cells) > 1:
                        desc = cells[1].get_text(strip=True)[:300]

                    assessments.append({
                        "name": name,
                        "url": full_url,
                        "test_type": primary_type,
                        "test_type_label": TEST_TYPE_MAP.get(primary_type, primary_type),
                        "all_types": type_codes if type_codes else [primary_type],
                        "description": desc,
                        "duration": "",
                        "remote_testing": True,
                        "adaptive": False,
                    })
                    logger.info(f"  Found: {name}")

                except Exception as e:
                    logger.debug(f"Row error: {e}")
                    continue

            if not found_new:
                break

            start += 12

    return assessments


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting SHL catalog scrape...")
    assessments = scrape_all()
    logger.info(f"Total assessments found: {len(assessments)}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(assessments, f, indent=2)
    logger.info(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()