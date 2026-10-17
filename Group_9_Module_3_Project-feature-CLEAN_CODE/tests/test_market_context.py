from __future__ import annotations

import unittest

import pandas as pd

from src.market_context import parse_cape_table, parse_news_feed, summarize_sp500_valuation


class MarketContextTests(unittest.TestCase):
    def test_parse_cape_table_extracts_dates_and_values(self) -> None:
        html = """
        <table id="datatable">
          <tr><th>Date</th><th>Value</th></tr>
          <tr><td>Jan 1, 2025</td><td>&#x2002; 30.5</td></tr>
          <tr><td>Dec 1, 2024</td><td>29.0</td></tr>
        </table>
        """

        result = parse_cape_table(html)

        self.assertEqual(result["CAPE"].tolist(), [29.0, 30.5])
        self.assertEqual(result.iloc[-1]["Date"], pd.Timestamp("2025-01-01"))

    def test_valuation_compares_latest_cape_to_preceding_history(self) -> None:
        history = pd.DataFrame(
            {
                "Date": ["2018-01-01", "2019-01-01", "2025-01-01"],
                "CAPE": [10.0, 20.0, 40.0],
            }
        )

        result = summarize_sp500_valuation(history)

        self.assertEqual(
            result["status"],
            "Relatively overvalued vs historical average",
        )
        self.assertEqual(result["historical_average"], 15.0)
        self.assertAlmostEqual(result["difference_percent"], 166.6666666667)

    def test_news_parser_only_returns_https_articles(self) -> None:
        feed = """<rss><channel>
          <item>
            <title>SP 500 market update</title>
            <link>https://example.com/article</link>
            <source>Example News</source>
            <pubDate>Sat, 17 Oct 2026 12:00:00 GMT</pubDate>
          </item>
          <item>
            <title>Unsafe link</title>
            <link>javascript:alert(1)</link>
          </item>
        </channel></rss>"""

        result = parse_news_feed(feed)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["Publisher"], "Example News")
        self.assertEqual(result[0]["Article"], "https://example.com/article")
        self.assertIn("2026-10-17", result[0]["Published"])


if __name__ == "__main__":
    unittest.main()
