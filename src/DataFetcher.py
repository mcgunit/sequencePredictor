import os, sys, csv, time, ast, re
from datetime import datetime
from html import unescape
import requests


def zodiac_codec():
    """
    (encode_zodiac, decode_zodiac) from Helpers, imported lazily: Helpers
    pulls in optuna/sklearn/scipy, which the fetcher only needs for Joker+
    draws (the API appends the English sign name to the digits, the CSVs
    carry the Dutch spelling - the codec is the single source of truth for
    that mapping). Callers run from the repo root import Helpers as
    `src.Helpers`; a standalone `python src/DataFetcher.py` run only has the
    src directory on sys.path, hence the fallback.
    """
    try:
        from src.Helpers import encode_zodiac, decode_zodiac
    except ImportError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from Helpers import encode_zodiac, decode_zodiac
    return encode_zodiac, decode_zodiac


def prize_tables():
    """src/PrizeTables, imported the way zodiac_codec imports Helpers (from the repo root or from within src/)."""
    try:
        from src import PrizeTables
    except ImportError:
        import PrizeTables
    return PrizeTables



# What a reply that is not JSON says, for the log. The API's failures of 9 and 16 Nov 2026 were web pages (HTTP 200),
# and the first 160 characters of a page are its doctype and style sheet: nothing of what it says. So a page is
# reduced to its title and its visible text (scripts, styles and tags left out, white space collapsed); any other
# reply is kept as it is. At most `limit` characters.
PAGE_TEXT_LIMIT = 400

def page_text(body, limit=PAGE_TEXT_LIMIT):
    text = str(body or "")
    if re.search(r"<\s*(!doctype|html|head|body|title)\b", text, re.I):
        title = re.search(r"<title[^>]*>(.*?)</title\s*>", text, re.I | re.S)
        visible = re.sub(r"<(script|style|noscript|head|svg)\b.*?</\1\s*>", " ", text, flags=re.I | re.S)
        # a page cut off inside its style sheet or script (no closing tag) has no visible text after it
        visible = re.sub(r"<(script|style|noscript|svg)\b.*\Z", " ", visible, flags=re.I | re.S)
        visible = re.sub(r"<!--.*?-->", " ", visible, flags=re.S)
        visible = re.sub(r"<[^>]*>", " ", visible)
        parts = []
        if title:
            parts.append("title: " + re.sub(r"<[^>]*>", " ", title.group(1)))
        parts.append("text: " + visible)
        text = unescape("; ".join(parts))
        text = re.sub(r"\s+", " ", text).strip()
        if text in ("text:", "title: ; text:"):
            text = "a web page without visible text"
    else:
        text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."



# A run that fetched nothing is a failed run (16 Nov 2026): when the fetch failed for every lottery game the run
# asked for, the predictor ends with this status instead of 0, and the scheduler (jobs.js) starts it again an hour
# later. Not for a run where at least one game answered: its failed games are a draw behind and catch up on the next
# run. 74 is sysexits' EX_IOERR; 75 is taken by a control that could not start on the CPU worker.
NO_DATA_EXIT = 74

def no_data_status(attempted, failed):
    """NO_DATA_EXIT when every game whose draws were asked for failed (and at least one was asked), else 0."""
    attempted, failed = list(attempted), set(failed)
    return NO_DATA_EXIT if attempted and all(g in failed for g in attempted) else 0


class DataFetcher():
    startDate = int(time.time() - (2*24*3600)) * 1000 #((current time in seconds) - (days in seconds)) * 1000 for ms
    endDate = int(time.time()) * 1000

    # Joker+ draws carry exactly six digits in front of the sign name.
    JOKERPLUS_DIGIT_COUNT = 6

    def parse_primary_string(self, primary_string):
        """
        (digits, sign) of a single-string "primary" result. Pick3 publishes
        its draw as one string of digits ('998'), Joker+ as the six digits
        immediately followed by the ENGLISH zodiac name ('430109Scorpio').
        Digits are split per character so leading zeros survive ('000123'
        -> [0, 0, 0, 1, 2, 3]); the trailing alphabetic part is the sign
        (None when absent - Pick3). Anything that is not "digits then
        letters" falls back to the historical per-character int conversion,
        so the other games' parsing is unchanged (including how it fails).
        """
        text = str(primary_string).strip()
        match = re.fullmatch(r"(\d+)([A-Za-z]*)", text)
        if not match:
            return [int(char) for char in text], None
        digits = [int(char) for char in match.group(1)]
        sign = match.group(2) or None
        return digits, sign

    def parse_date(self, date_string):
        if "00:00:00.0000000" in date_string:
            return datetime.strptime(date_string.split(" ")[0], '%Y-%m-%d')
        else:
            return datetime.strptime(date_string, '%Y-%m-%d')

    def calculate_start_date(self, filePath):
        csv_file_path = filePath
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)  # Read all rows into a list
                if rows:
                    latest_date_str = rows[1][0].split(";")[0]  # Get the date from the second row
                    #print("Latest date: ", latest_date_str)
                    latest_date = self.parse_date(latest_date_str)
                    today = datetime.now()
                    days_ago = (today - latest_date).days
                   
                    start_date = int(datetime.now().timestamp() - (days_ago * 24 * 3600)) * 1000
                    
                    return start_date
                else:
                    # File is empty, so fetch data for the last 30 days
                    return int(datetime.now().timestamp() - (30 * 24 * 3600)) * 1000
        except FileNotFoundError:
            # File doesn't exist, so fetch data for the last 6 days
            return int(datetime.now().timestamp() - (30 * 24 * 3600)) * 1000

    # A failed answer is tried again before the caller falls back to the CSV on disk: on 9 Nov 2026, at 00:30 UTC, the
    # API answered every game with a reply that was not JSON, and the day's lottery predictions were skipped (the run
    # moved to 06:00 Brussels). Each failure is logged with its HTTP status and the start of the reply, so the next one
    # says what the API sent - for a web page its title and visible text (page_text). RETRY_WAITS: the seconds before
    # the second and third attempt.
    RETRY_WAITS = (30, 120)

    def fetch_json(self, url, headers, game, sleep=time.sleep, get=None):
        get = get or requests.get
        waits = list(self.RETRY_WAITS)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = get(url=url, headers=headers, timeout=(10, 120))
                try:
                    return response.json()
                except ValueError as e:
                    body = page_text(getattr(response, "text", "") or "")
                    raise ValueError(f"HTTP {getattr(response, 'status_code', '?')}, not JSON ({e}): {body!r}")
            except Exception as e:
                if not waits:
                    raise
                wait = waits.pop(0)
                print(f"Data fetch for {game} failed (attempt {attempt}): {e} - trying again in {wait} s")
                sleep(wait)

    def getLatestData(self, game, filePath, dryRun=False):
        if not game:
            # game-names= empty answers with EVERY game's draws, which would
            # be appended to whatever file was passed (found in review with
            # the market games, whose history is not the lottery's to fetch)
            raise ValueError("getLatestData needs a game name - an empty one would fetch every game's draws")

        print("Startdate: ", self.startDate, datetime.fromtimestamp(self.startDate/1000).strftime("%A, %B %d, %Y %I:%M:%S"))
        print("Enddate: ", self.endDate, datetime.fromtimestamp(self.endDate/1000).strftime("%A, %B %d, %Y %I:%M:%S"))
        url = f"https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&date-from={self.startDate}&size=62&date-to={self.endDate}&game-names={game}"
        #url = https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&date-from=1751328000000&size=62&date-to=1756684800000&game-names=Pick3
        #print("url: ", url)
        
        headers = {
            "User-Agent": "wget/1.21.4",
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Connection": "Keep-Alive"
        }
        # Connect/read timeouts: without them a stalled API hangs the caller
        # forever - for the daily Predictor that means holding process.lock
        # indefinitely (observed 2026-09-08: a tuning run sat 9+ minutes in
        # this call). A timeout raises, and every caller falls back to the
        # CSV already on disk.
        data = self.fetch_json(url, headers, game)
        
        draws = data.get("draws", [])

        # Lotto, EuroMillions and EuroDreams: the same answer carries each
        # draw's prize tiers - kept next to the CSV for the profit of their
        # tickets (src/PrizeTables.py, 3 Nov 2026). A table that cannot be
        # kept costs the profit of that draw, never its numbers.
        prizeGame = prize_tables().game_of(os.path.basename(os.path.dirname(filePath)))
        if prizeGame and not dryRun:
            try:
                prize_tables().record(os.path.dirname(filePath), prizeGame, draws)
            except Exception as e:
                print(f"Prize tables for {prizeGame} could not be kept: {e}")

        rows = []
        for draw in draws:
            draw_date = datetime.utcfromtimestamp(draw["drawTime"] / 1000).strftime("%Y-%m-%d")

            # Initialize
            primary = []
            bonus = None
            malformed = False

            # Look through results to separate primary and bonus
            for result in draw.get("results", []):
                
                if result.get("drawType") == "normal":
                    primary_raw = result.get("primary", [])
                    if len(primary_raw) == 1:
                        primary, sign = self.parse_primary_string(primary_raw[0])
                        if sign is not None:
                            # Joker+: the sign is stored in the CSV as its
                            # Dutch spelling (the 7th field, after the six
                            # digits) - encoding the API's English name and
                            # decoding it back yields exactly that spelling,
                            # so the row dedupes against the historical
                            # exports. An unrecognized sign or a wrong digit
                            # count is a malformed draw: skip it (the next
                            # run retries from the latest CSV date) rather
                            # than write a row every loader would reject.
                            encode_zodiac, decode_zodiac = zodiac_codec()
                            try:
                                if len(primary) != self.JOKERPLUS_DIGIT_COUNT:
                                    raise ValueError(f"expected {self.JOKERPLUS_DIGIT_COUNT} digits, got {len(primary)}")
                                primary = primary + [decode_zodiac(encode_zodiac(sign))]
                            except ValueError as ve:
                                print(f"Skipping {draw_date}: malformed Joker+ draw '{primary_raw[0]}' ({ve})")
                                primary = []
                                malformed = True
                    else: 
                        primary = [int(n) for n in result.get("primary", [])]

                    if result.get("secondary", []):
                        bonus = [int(n) for n in result.get("secondary", [])]
                
                elif result.get("drawType") == "bonus":
                    bonus_list = result.get("primary", [])
                    if bonus_list:
                        bonus = [int(n) for n in bonus_list]

            # Skip draws with no published numbers yet (e.g. today's draw is
            # scheduled but hasn't happened/been published) - writing a
            # date-only row (no numbers) corrupts the CSV's column count for
            # every downstream numpy.genfromtxt load.
            if not primary:
                if not malformed:
                    print(f"Skipping {draw_date}: no numbers published yet")
                continue

            # Compose the line
            numbers_string = ";".join(map(str, primary))
            if bonus:
                numbers_string = ";".join(map(str, primary + bonus))
            rows.append(f"{draw_date};{numbers_string}")
            print(f"{draw_date};{numbers_string}")
        
            
        # CSV File Handling
        csv_file_path = filePath
        existing_rows = []
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    existing_rows.append(row[0])  # Assuming the first column contains the data you want to check for duplicates
        except FileNotFoundError:
            # File doesn't exist, so create it (with header if needed)
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                #writer.writerow(['Date;Numbers'])  # Optional: Write a header row
        
        new_rows = []
        for row in rows:
            if row not in existing_rows:
                new_rows.append(row)
                existing_rows.append(row)  # Update existing_rows to prevent future duplicates
        
        #print(existing_rows)

        # Extract the header
        header = existing_rows[0]

        # Extract the data rows
        data_rows = existing_rows[1:]

        #print("data rows: ", data_rows)

        # Sort the data rows by date in descending order
        sorted_data_rows = sorted(data_rows, key=lambda x: self.parse_date(x.split(';')[0]), reverse=True)

        #print("sorted data: ", sorted_data_rows)

        # Put the header back at the beginning
        sorted_data = [header] + sorted_data_rows

        #Print the sorted data
        # for row in sorted_data:
        #     print(row)
        
        # Write the sorted data back to the CSV file
        if not dryRun:
            with open(csv_file_path, 'w', newline='') as csvfile:  # 'w' for write mode
                writer = csv.writer(csvfile, delimiter=";")
                for row in sorted_data:
                    writer.writerow(row.split(';'))  # Split the row into a list of values



def _self_check():
    # page_text: a page is its title and visible text, never its style sheet
    page = ('<!doctype html> <html lang="en"> <head><meta charset="utf-8"><title>Site  Maintenance</title>'
            '<style>body { font: 20px Helvetica } h1 { font-size: 50px }</style><script>var a = "<b>x</b>";</script></head>'
            '<body><!-- hidden --><h1>We&rsquo;ll be back soon!</h1><p>Sorry for the\n inconvenience &amp; thanks.</p></body></html>')
    text = page_text(page)
    assert text == "title: Site Maintenance; text: We\u2019ll be back soon! Sorry for the inconvenience & thanks.", text
    assert "font" not in text and "var a" not in text and "hidden" not in text
    # the reply of 16 Nov as the old log kept it: cut off inside the style sheet
    assert page_text('<!doctype html> <html lang="en"> <head> <style> body { text-align: center') == "a web page without visible text"
    assert page_text('<html><head><title>502 Bad Gateway</title></head><body><center>nginx</center></body></html>') == "title: 502 Bad Gateway; text: nginx"
    assert page_text("upstream  connect error\n or disconnect") == "upstream connect error or disconnect"
    assert page_text("") == "" and page_text(None) == ""
    long = page_text("<html><body>" + "word " * 500 + "</body></html>")
    assert len(long) == PAGE_TEXT_LIMIT and long.endswith("...")

    # fetch_json: three attempts, and the error of the last one carries the page's text
    class Reply:
        status_code = 200
        def __init__(self, text): self.text = text
        def json(self):
            import json
            return json.loads(self.text)
    fetcher = DataFetcher()
    waits, calls = [], []
    def bad(url, headers, timeout):
        calls.append(url)
        return Reply(page)
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()) as out:
            fetcher.fetch_json("u", {}, "Lotto", sleep=waits.append, get=bad)
        raise AssertionError("a page must not pass as JSON")
    except ValueError as e:
        assert "HTTP 200, not JSON" in str(e) and "Site Maintenance" in str(e) and "back soon" in str(e), str(e)
        assert "font" not in str(e)
    assert len(calls) == 3 and waits == list(DataFetcher.RETRY_WAITS), (calls, waits)
    assert out.getvalue().count("trying again") == 2 and "Site Maintenance" in out.getvalue()
    answers = [Reply(page), Reply('{"draws": []}')]
    with contextlib.redirect_stdout(io.StringIO()):
        assert fetcher.fetch_json("u", {}, "Lotto", sleep=lambda s: None, get=lambda url, headers, timeout: answers.pop(0)) == {"draws": []}

    # no_data_status: only a run where every asked game failed
    assert no_data_status(["lotto", "keno"], ["lotto", "keno"]) == NO_DATA_EXIT == 74
    assert no_data_status(["lotto", "keno"], ["lotto"]) == 0
    assert no_data_status(["lotto"], []) == 0
    assert no_data_status([], []) == 0          # a market-only run asks the lottery API nothing
    print("DataFetcher self-check OK")


if __name__ == "__main__":
    dataFetcher = DataFetcher()
    print("Running datafetcher")
    #print("Checking date range: ", dataFetcher.startDate, "-", dataFetcher.endDate)
    current_year = datetime.now().year
    path = os.getcwd()
    game = "keno"
    dataPath = os.path.join(path, "data", "trainingData", game.lower())
    file = f"{game.lower()}-gamedata-NL-{current_year}.csv"
    filePath = os.path.join(dataPath, file)
    print("File path: ", filePath)
    dataFetcher.startDate = dataFetcher.calculate_start_date(filePath)
    #print("Startdate: ", dataFetcher.startDate, datetime.fromtimestamp(dataFetcher.startDate/1000).strftime("%A, %B %d, %Y %I:%M:%S"))
    dataFetcher.getLatestData(game, filePath, dryRun=True)

