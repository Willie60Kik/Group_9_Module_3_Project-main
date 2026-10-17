from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import pandas as pd
import streamlit as st


CAPE_URL = "https://www.multpl.com/shiller-pe/table/by-month"
NEWS_RSS_URL = "https://news.google.com/rss/search?"
REQUEST_TIMEOUT_SECONDS = 10


class _CapeTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_cell = False
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []
        self.cell_text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "datatable":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.current_row = []
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_text = []

    def handle_data(self, data: str) -> None:
        if self.in_table and self.in_cell:
            self.cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and tag in {"th", "td"} and self.in_cell:
            self.current_row.append(" ".join("".join(self.cell_text).split()))
            self.in_cell = False
        elif self.in_table and tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
        elif tag == "table" and self.in_table:
            self.in_table = False


def parse_cape_table(html: str) -> pd.DataFrame:
    parser = _CapeTableParser()
    parser.feed(html)
    if not parser.rows or parser.rows[0][:2] != ["Date", "Value"]:
        raise ValueError("The valuation source did not contain the expected CAPE table.")

    values = pd.DataFrame(parser.rows[1:], columns=["Date", "CAPE"])
    values["Date"] = pd.to_datetime(
        values["Date"], format="%b %d, %Y", errors="coerce"
    )
    values["CAPE"] = pd.to_numeric(values["CAPE"], errors="coerce")
    values = (
        values.dropna(subset=["Date", "CAPE"])
        .loc[lambda frame: frame["CAPE"] > 0]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if values.empty:
        raise ValueError("The valuation source returned no valid CAPE observations.")
    return values


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_sp500_cape_history() -> pd.DataFrame:
    request = Request(
        CAPE_URL,
        headers={"User-Agent": "SP500ResearchDashboard/1.0"},
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        html = response.read().decode("utf-8", errors="replace")
    return parse_cape_table(html)


def summarize_sp500_valuation(history: pd.DataFrame) -> dict[str, object]:
    required = {"Date", "CAPE"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            "CAPE history is missing required columns: "
            + ", ".join(sorted(missing))
        )

    observations = history.copy()
    observations["Date"] = pd.to_datetime(observations["Date"], errors="coerce")
    observations["CAPE"] = pd.to_numeric(observations["CAPE"], errors="coerce")
    observations = observations.dropna(subset=["Date", "CAPE"]).sort_values("Date")
    if observations.empty:
        raise ValueError("CAPE history has no usable observations.")
    latest = observations.iloc[-1]
    latest_date = pd.Timestamp(latest["Date"])
    reference_end = latest_date - pd.DateOffset(years=1)
    reference = observations.loc[observations["Date"] < reference_end, "CAPE"]
    if reference.empty:
        raise ValueError("Not enough historical CAPE data to calculate a baseline.")

    historical_average = float(reference.mean())
    current_cape = float(latest["CAPE"])
    difference_percent = (current_cape / historical_average - 1) * 100
    if current_cape > historical_average:
        status = "Relatively overvalued vs historical average"
    elif current_cape < historical_average:
        status = "Relatively undervalued vs historical average"
    else:
        status = "At the historical average"
    return {
        "status": status,
        "current_cape": current_cape,
        "historical_average": historical_average,
        "difference_percent": difference_percent,
        "as_of": latest_date,
        "reference_start": pd.Timestamp(observations.iloc[0]["Date"]),
        "reference_end": reference_end,
    }


def _format_published_date(value: str) -> str:
    if not value:
        return "Date unavailable"
    try:
        published: datetime = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return value
    return published.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def parse_news_feed(xml_text: str, limit: int = 5) -> list[dict[str, str]]:
    if limit < 1:
        raise ValueError("News article limit must be positive.")
    root = ElementTree.fromstring(xml_text)
    articles = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        parsed_link = urlparse(link)
        if not title or parsed_link.scheme != "https" or not parsed_link.netloc:
            continue
        source = (item.findtext("source") or parsed_link.netloc).strip()
        published = (item.findtext("pubDate") or "").strip()
        articles.append(
            {
                "Headline": title,
                "Publisher": source,
                "Published": _format_published_date(published),
                "Article": link,
            }
        )
        if len(articles) == limit:
            break
    return articles


@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_sp500_news(limit: int = 5) -> list[dict[str, str]]:
    query = urlencode(
        {
            "q": "S&P 500 OR SPY market",
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
    )
    request = Request(
        NEWS_RSS_URL + query,
        headers={"User-Agent": "SP500ResearchDashboard/1.0"},
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        xml_text = response.read().decode("utf-8", errors="replace")
    return parse_news_feed(xml_text, limit)


def show_market_context(
    decision_basis: str,
    signal_as_of: pd.Timestamp | None = None,
) -> None:
    st.subheader("Valuation and news context")
    st.caption(
        f"Signal basis: {decision_basis}. News headlines are context only and "
        "do not change the BUY / SELL / HOLD decision."
    )
    if signal_as_of is not None:
        st.caption(
            f"Signal data date: {pd.Timestamp(signal_as_of).date()}. Valuation "
            "and news have independent publication dates; later context was not "
            "available to a historical signal."
        )
    valuation_column, news_column = st.columns([1, 2])

    with valuation_column:
        st.markdown("**S&P 500 CAPE valuation**")
        try:
            valuation = summarize_sp500_valuation(load_sp500_cape_history())
        except (OSError, TimeoutError, URLError, ValueError) as exc:
            st.warning(f"Valuation data unavailable: {exc}")
        else:
            st.metric(
                valuation["status"],
                f"{valuation['current_cape']:.2f}",
                delta=f"{valuation['difference_percent']:+.1f}% vs historical average",
                border=True,
            )
            st.caption(
                f"As of {valuation['as_of'].date()}. Historical mean "
                f"{valuation['historical_average']:.2f}; reference data begins "
                f"{valuation['reference_start'].date()} and excludes the latest "
                "12 months."
            )
            st.caption(
                "Relative valuation only—not an intrinsic fair-value estimate "
                "or a trading signal. [Source: Multpl / Robert Shiller](https://www.multpl.com/shiller-pe/table/by-month)."
            )

    with news_column:
        st.markdown("**Recent S&P 500 / SPY headlines**")
        try:
            articles = load_sp500_news()
        except (OSError, TimeoutError, URLError, ElementTree.ParseError, ValueError) as exc:
            st.warning(f"News feed unavailable: {exc}")
        else:
            if articles:
                st.dataframe(
                    pd.DataFrame(articles),
                    hide_index=True,
                    column_config={
                        "Article": st.column_config.LinkColumn(
                            "Article", display_text="Open"
                        )
                    },
                    width="stretch",
                )
            else:
                st.info("No recent headlines were returned by the news feed.")
            st.caption(
                "Google News RSS search results; headlines may be incomplete or "
                "delayed. Open the article to review its context."
            )
