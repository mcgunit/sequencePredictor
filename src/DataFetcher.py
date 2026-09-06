import os, sys, csv, time, ast, re
from datetime import datetime
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

    def getLatestData(self, game, filePath, dryRun=False):

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
        response = requests.get(url=url, headers=headers)
        
        #print("response: ", response.json())
        data = response.json()
        
        draws = data.get("draws", [])
        
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

