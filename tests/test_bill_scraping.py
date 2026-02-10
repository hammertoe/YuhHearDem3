# Tests for bill scraping and entity extraction.
import pytest
from unittest.mock import Mock, patch

from lib.scraping.bill_scraper import BillScraper
from lib.processors.bill_entity_extractor import BillEntityExtractor


@pytest.fixture
def sample_bill_data():
    """Sample bill data for testing."""
    return [
        {
            "bill_number": "HR 1234",
            "title": "Road Traffic (Amendment) Bill",
            "description": "An Act to amend the Road Traffic Act...",
            "status": "Introduced",
            "introduced_date": "2026-01-15",
            "source_url": "https://www.parliament.gov.bb/bill/HR1234",
            "source_text": "THE ROAD TRAFFIC ACT\n\nBE IT ENACTED...",
            "category": "Transport",
        },
        {
            "bill_number": "SB 5678",
            "title": "Health Services Improvement Act",
            "description": "An Act to improve healthcare services...",
            "status": "Passed",
            "passed_date": "2026-02-20",
            "source_url": "https://www.parliament.gov.bb/bill/SB5678",
            "source_text": "THE HEALTH SERVICES ACT\n\nBE IT ENACTED...",
            "category": "Health",
        },
    ]


def test_bill_scraper_init():
    """Test bill scraper initialization."""
    scraper = BillScraper()
    assert scraper.base_url == "https://www.parliament.gov.bb"
    assert scraper.rate_limit_delay == 1.0
    print("✅ Bill scraper initialization works")


def test_is_bill_url():
    """Test bill URL detection."""
    scraper = BillScraper()

    assert scraper._is_bill_url("/bill/HR1234") == True
    assert scraper._is_bill_url("/legislation/HR1234") == True
    assert scraper._is_bill_url("/about") == False
    assert scraper._is_bill_url("/contact") == False
    print("✅ Bill URL detection works")


def test_resolve_url():
    """Test URL resolution."""
    scraper = BillScraper()

    assert (
        scraper._resolve_url("/bill/HR1234")
        == "https://www.parliament.gov.bb/bill/HR1234"
    )
    assert (
        scraper._resolve_url("https://example.com/bill/HR1234")
        == "https://example.com/bill/HR1234"
    )
    assert (
        scraper._resolve_url("relative/path")
        == "https://www.parliament.gov.bb/relative/path"
    )
    print("✅ URL resolution works")


def test_parse_bill_number_from_url():
    """Test bill number parsing from URL."""
    from lib.scraping.bill_scraper import BillScraper

    scraper = BillScraper()

    test_cases = [
        ("/bill/HR1234", "HR1234"),
        ("/bill/BILL-42", "BILL-42"),
        ("/legislation/SB5678", "SB5678"),
        ("/cap-295", "Cap 295"),
    ]

    for url, expected in test_cases:
        result = scraper._parse_bill_number(Mock(), url)
        assert result.get("bill_number") == expected, f"Failed for {url}"

    print("✅ Bill number parsing works")


def test_entity_extractor_init():
    """Test entity extractor initialization."""
    extractor = BillEntityExtractor()
    assert extractor.nlp is not None
    print("✅ Entity extractor initialization works")


def test_extract_entities_from_bill():
    """Test entity extraction from bill text."""
    extractor = BillEntityExtractor()

    bill_data = {
        "title": "Road Traffic (Amendment) Bill",
        "description": "This bill amends the Road Traffic Act to increase penalties for speeding offenses. It also introduces new provisions for distracted driving violations.",
        "source_text": "THE ROAD TRAFFIC ACT\n\nBE IT ENACTED...",
    }

    result = extractor.extract_entities_from_bill(bill_data)

    assert "extracted_entities" in result
    entities = result["extracted_entities"]

    assert "organizations" in entities
    assert "topics" in entities
    assert "persons" in entities

    print(f"✅ Entity extraction works: {len(entities.get('topics', []))} topics")


def test_extract_category():
    """Test bill categorization."""
    extractor = BillEntityExtractor()

    transport_bill = {
        "title": "Road Traffic (Amendment) Bill",
        "description": "This bill addresses traffic safety and vehicle regulations.",
        "extracted_entities": {
            "organizations": [{"text": "Ministry of Transport"}],
            "topics": [{"text": "traffic safety"}],
        },
    }

    category = extractor.extract_category(transport_bill)
    assert category == "Transport", f"Expected 'Transport', got {category}"

    health_bill = {
        "title": "Health Services Improvement Act",
        "description": "This bill improves healthcare services and hospital funding.",
        "extracted_entities": {
            "organizations": [{"text": "Ministry of Health"}],
            "topics": [{"text": "healthcare services"}],
        },
    }

    category = extractor.extract_category(health_bill)
    assert category == "Health", f"Expected 'Health', got {category}"

    general_bill = {
        "title": "General Services Act",
        "description": "This bill provides for various administrative services.",
        "extracted_entities": {},
    }

    category = extractor.extract_category(general_bill)
    assert category == "General", f"Expected 'General', got {category}"

    print("✅ Bill categorization works")


def test_generate_keywords():
    """Test keyword generation."""
    extractor = BillEntityExtractor()

    bill_data = {
        "title": "Road Traffic (Amendment) Bill",
        "description": "This bill addresses traffic safety.",
        "extracted_entities": {
            "organizations": [
                {"text": "Ministry of Transport"},
                {"text": "Road Traffic Authority"},
            ],
            "topics": [
                {"text": "traffic safety"},
                {"text": "speeding"},
                {"text": "vehicle regulations"},
            ],
        },
    }

    keywords = extractor._generate_keywords(bill_data)

    assert len(keywords) > 0
    assert "traffic" in keywords
    assert "safety" in keywords
    assert "road" in keywords
    assert "transport" in keywords

    print(f"✅ Keyword generation works: {len(keywords)} keywords")


def test_process_bills():
    """Test processing multiple bills."""
    extractor = BillEntityExtractor()

    bills = [
        {
            "bill_number": "HR 1234",
            "title": "Road Traffic (Amendment) Bill",
            "description": "Amends Road Traffic Act.",
        },
        {
            "bill_number": "SB 5678",
            "title": "Health Services Act",
            "description": "Improves healthcare services.",
        },
    ]

    processed = extractor.process_bills(bills)

    assert len(processed) == 2
    assert all("id" in bill for bill in processed)
    assert all("category" in bill for bill in processed)
    assert all("keywords" in bill for bill in processed)
    assert all("extracted_entities" in bill for bill in processed)

    print("✅ Batch bill processing works")


@patch("lib.scraping.bill_scraper.requests.Session.get")
def test_scrape_bill_page(mock_get):
    """Test scraping a bill page."""
    mock_response = Mock()
    mock_response.text = """
        <html>
            <h1 class="bill-title">Road Traffic (Amendment) Bill, 2025</h1>
            <strong>Introduced: 2025-01-15</strong>
            <div class="description">
                This bill amends the Road Traffic Act to increase penalties.
            </div>
        </html>
    """
    mock_response.raise_for_status = Mock()

    mock_get.return_value = mock_response

    from lib.scraping.bill_scraper import BillScraper

    scraper = BillScraper()

    bill_data = scraper.scrape_bill("https://www.parliament.gov.bb/bill/HR1234")

    assert bill_data is not None
    assert "title" in bill_data
    assert "scraped_at" in bill_data

    print("✅ Bill page scraping works")


def test_scrape_all_bills():
    """Test scraping all bills with limit."""
    with patch("lib.scraping.bill_scraper.BillScraper.fetch_page") as mock_fetch:
        mock_fetch.return_value = """
            <html>
                <a href="/bill/HR1234">Road Traffic Bill</a>
                <a href="/bill/SB5678">Health Bill</a>
            </html>
        """

        from lib.scraping.bill_scraper import BillScraper

        scraper = BillScraper()

        bills = scraper.scrape_all_bills(max_bills=2)

        assert len(bills) == 2
        assert all("source_url" in bill for bill in bills)

        print("✅ Batch bill scraping works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
