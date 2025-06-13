import os, csv, time, ast
from datetime import datetime
import requests

class DataFetcher():
    startDate = int(time.time() - (2*24*3600)) * 1000 #((current time in seconds) - (days in seconds)) * 1000 for ms
    endDate = int(time.time()) * 1000

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
                   
                    start_date = int(datetime.now().timestamp() - (days_ago * 24 * 3600))*1000
                    
                    return start_date
                else:
                    # File is empty, so fetch data for the last 30 days
                    return int(datetime.now().timestamp() - (30 * 24 * 3600))
        except FileNotFoundError:
            # File doesn't exist, so fetch data for the last 6 days
            return int(datetime.now().timestamp() - (30 * 24 * 3600))

    def getLatestData(self, game, filePath):
        url = f"https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&date-from={self.startDate}&size=62&date-to={self.endDate}&game-names={game}"
        #url = "https://apim.prd.natlot.be/api/v4/draw-games/draws?status=PAYABLE&date-from=1746057600000&size=62&date-to=1751414400000&game-names=Keno"
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
            numbers = [str(result["primary"]) for result in draw.get("results", []) if "primary" in result]
            numbersArray = ast.literal_eval(numbers[0])
            numbers_string = ";".join(map(str, numbersArray))  # Join the numbers with semicolons
            rows.append(f"{draw_date};{numbers_string}")
            print(f"{draw_date};{numbers_string}") # Print in the desired format
        
            
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
    game = "Keno"
    dataPath = os.path.join(path, "data", "trainingData", game.lower())
    file = f"{game.lower()}-gamedata-NL-{current_year}.csv"
    filePath = os.path.join(dataPath, file)
    print("File path: ", filePath)
    dataFetcher.startDate = dataFetcher.calculate_start_date(filePath)
    print("Startdate: ", dataFetcher.startDate, datetime.fromtimestamp(dataFetcher.startDate/1000).strftime("%A, %B %d, %Y %I:%M:%S"))
    dataFetcher.getLatestData(game, filePath)

