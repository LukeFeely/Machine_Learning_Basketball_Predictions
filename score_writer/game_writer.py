import csv
import os
from pathlib import Path


class GameWriter:
    """
    Writes data for games to output csv as series of features
    """

    def __init__(self, output_path, games_list, append=False):
        """
        :param output_path: specify the output path for the csv file
        :param games_list: list of games to write (each game gets its own row)
        :param append: open in append mode
        """
        self.output_path = Path(output_path)
        self.games_list = games_list
        self.should_append_to_file = append

    def write(self):
        """
        Writes all games in self.games_list to the csv

        :return: None
        """
        headers = self.games_list[0].keys()  # get the list of csv headers
        file_exists = self.output_path.exists()
        if file_exists and not self.should_append_to_file:  # remove file if exists and not append
            os.remove(self.output_path)
        # writes games (which are dictionaries) to the csv using csv.DictWriter
        with self.output_path.open("a") as games_csv:
            csv_writer = csv.DictWriter(games_csv, fieldnames=headers, lineterminator='\n')
            if not file_exists:
                csv_writer.writeheader()  # only add the headers once
            for game in self.games_list:
                csv_writer.writerow(game)
