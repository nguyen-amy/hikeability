import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path so we can import the scrapers module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrapers.wta_daily_scraper import WTADailyScraper

class TestWTADailyScraper(unittest.TestCase):

    @patch('scrapers.wta_daily_scraper.storage.Client')
    def setUp(self, mock_gcs_client):
        """
        setUp runs before every test. We patch the GCS Client so 
        instantiating the scraper doesn't attempt to authenticate with Google.
        """
        self.scraper = WTADailyScraper(bucket_name="fake-test-bucket")

    def test_sort_reports_chronologically(self):
        """Test that the sorting function correctly handles dates, including invalid ones."""
        sample_reports = [
            {'title': 'Recent Hike', 'date_hiked': 'Apr 19, 2026'},
            {'title': 'Old Hike', 'date_hiked': 'Jan 10, 2026'},
            {'title': 'Mid Hike', 'date_hiked': 'Feb 15, 2026'},
            {'title': 'Invalid Date Hike', 'date_hiked': 'Not a date'} # Should fall back to datetime.min
        ]
        
        sorted_reports = self.scraper.sort_reports_chronologically(sample_reports)
        
        # Check order: Invalid (min date) -> Jan -> Feb -> Apr
        self.assertEqual(sorted_reports[0]['title'], 'Invalid Date Hike')
        self.assertEqual(sorted_reports[1]['title'], 'Old Hike')
        self.assertEqual(sorted_reports[2]['title'], 'Mid Hike')
        self.assertEqual(sorted_reports[3]['title'], 'Recent Hike')

    @patch('scrapers.wta_daily_scraper.requests.Session.get')
    def test_scrape_report_details(self, mock_get):
        """Test the HTML parsing logic with a mocked webpage response."""
        
        # 1. Create a fake HTML response that mimics WTA's structure
        mock_html = """
        <html>
            <h1 class="documentFirstHeading">Rattlesnake Ledge — Apr 19, 2026</h1>
            <a class="author">By TrailBlazer99</a>
            <div class="trip-condition"><h4>Trail</h4><span>Snow on trail</span></div>
            <div class="trip-condition"><h4>Bugs</h4><span>Terrible</span></div>
            <div id="tripreport-body"><p>It was a beautiful day.</p><p>Microspikes needed!</p></div>
            <a href="/go-hiking/hikes/rattlesnake-ledge">Rattlesnake Ledge</a>
        </html>
        """
        
        # 2. Configure our mocked requests.get to return this fake HTML
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html
        mock_get.return_value = mock_response
        
        # 3. Run the method
        result = self.scraper.scrape_report_details("https://fake-wta-url.com/report")
        
        # 4. Assert the Beautiful Soup parser extracted everything correctly
        self.assertIsNotNone(result)
        self.assertEqual(result['title'], 'Rattlesnake Ledge')
        self.assertEqual(result['date_hiked'], 'Apr 19, 2026')
        self.assertEqual(result['author'], 'TrailBlazer99')
        self.assertEqual(result['trail_conditions'], 'Snow on trail')
        self.assertEqual(result['bugs'], 'Terrible')
        self.assertEqual(result['report_text'], 'It was a beautiful day.\n\nMicrospikes needed!')
        
        # Check that the hike slug was extracted correctly
        self.assertEqual(len(result['associated_hikes']), 1)
        self.assertEqual(result['associated_hikes'][0]['hike_slug'], 'rattlesnake-ledge')

    @patch('scrapers.wta_daily_scraper.requests.Session.get')
    def test_scrape_report_details_404(self, mock_get):
        """Test that the scraper handles a failed HTTP request gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.scraper.scrape_report_details("https://fake-wta-url.com/report")
        
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()