import csv
import os
from datetime import date
from pathlib import Path

import pandas as pd

from Team.TeamStats import TeamStats
from score_writer.game import Game
from score_writer.game_writer import GameWriter
from score_writer.score_scraper import ScoreScraper

'''
Class used to stitch together data that has been scraped from basketball-reference.com
We adopted an approach of scraping the data season by season, saving each season to 
its own csv file and then using a method in this class to combine multiple seasons
together to form the training and test data 
'''


class CSVGenerator:
    def __init__(self, year_to_generate):
        self._year_to_generate = year_to_generate

    '''
    Takes a the csv file of raw data which we scraped for a particular season and transforms
    it into the features needed for the training
    '''

    def generate(self, data_frame=None, output_location=None, append=False):

        teams = pd.read_csv("data/teams.csv")[['TEAM_ID', 'ABBREVIATION']]
        filepath = f"data/game_stats/{self._year_to_generate}-{self._year_to_generate + 1}.csv"

        data_frame_defined = data_frame is not None
        games_frame = data_frame if data_frame_defined else pd.read_csv(filepath)

        games_list = []
        team_stats = TeamStats(teams['TEAM_ID'], self._year_to_generate)

        print("Generating data. This will take a minute....")
        num_rows = len(games_frame['date'])
        progress = 1
        ten_percent_data = int(num_rows / 10)
        next_progress_print = ten_percent_data
        for index, game_date in enumerate(games_frame['date']):
            home_team_id = games_frame['home_team_id'].iloc[index]
            away_team_id = games_frame['away_team_id'].iloc[index]

            # get the Team objects for the two teams in the game
            home_team = team_stats.get_team(home_team_id)
            away_team = team_stats.get_team(away_team_id)
            home_team_win = games_frame['is_home_winner'].iloc[index]

            home_team_points = games_frame['home_team_score'].iloc[index]
            away_team_points = games_frame['away_team_score'].iloc[index]

            # get the elo stats from the CSV - deprecated. We were unable to scrape
            # data for this from basketballreference.com
            home_team_elo = games_frame['home_team_elo'].iloc[index]
            away_team_elo = games_frame['away_team_elo'].iloc[index]
            home_team_raptor = games_frame['home_team_raptor'].iloc[index]
            away_team_raptor = games_frame['away_team_raptor'].iloc[index]

            home_team_hth_record = games_frame['home_team_hth_record'].iloc[index]
            away_team_hth_record = games_frame['away_team_hth_record'].iloc[index]

            # game object is an order dict which we can write directly to a CSV file. This represents the
            # feature which we are going to use for the model.
            current_game = Game(home_team, away_team, home_team_win, home_team_elo,
                                away_team_elo, home_team_raptor, away_team_raptor, home_team_hth_record,
                                away_team_hth_record)

            games_list.append(current_game)

            team_stats.record_game({"HOME_TEAM": home_team_id, "AWAY_TEAM": away_team_id, "RESULT": home_team_win,
                                    "HOME_TEAM_POINTS": home_team_points,
                                    "AWAY_TEAM_POINTS": away_team_points})
            # just for us so we can see the CSV being processed (its boring to wait)
            if index == next_progress_print:
                print(f"{progress * 10}%")
                progress += 1
                next_progress_print = ten_percent_data * progress
        output_file_name = (output_location if output_location is not None
                            else f"data/{self._year_to_generate}_games.csv")
        game_writer = GameWriter(output_file_name, games_list, append)
        game_writer.write()

    '''
    scrapes the data for a particular year and writes to a CSV 
    :param year_to_generate : (INT) you can specify the year that you want to scrape for as opposed using the class 
    attribute
    :param output_file_name : (STRING) you can specify where you would like the data to be written 
    :param should_overwrite_csv : (BOOL) you can specify whether you would like to append to the csv file (False) or 
    overwrite it (True - default) 
    :returns: void
    '''

    def generate_game_stats(self, year_to_generate=None, output_file_name=None, should_overwrite_csv=True):
        season_start_year = self._year_to_generate if year_to_generate is None else year_to_generate
        start_date = date.fromisoformat(f'{season_start_year}-10-01')
        if season_start_year == 2019:
            end_date = date.fromisoformat(f'{season_start_year + 1}-10-21')
        else:
            end_date = date.fromisoformat(f'{season_start_year + 1}-07-01')
        game_scraper = ScoreScraper(start_date, end_date)
        game_results = game_scraper.results_list

        output_file = (f"data/game_stats/{season_start_year}-{season_start_year + 1}.csv" if output_file_name is None
                       else output_file_name)
        is_file_existing = Path(output_file).is_file()

        if is_file_existing and should_overwrite_csv:
            os.remove(output_file)

        with open(output_file, 'a') as output_csv:
            '''
            iterate over all the games scraped by the programme
            '''
            for index, game_dict in enumerate(game_results):
                if index == 0:
                    headers = game_dict.keys()
                    writer = csv.DictWriter(output_csv, fieldnames=headers, lineterminator='\n')
                    writer.writeheader()
                writer.writerow(game_dict)

    '''
    scrapes the data for a given array of years. For the purposes of this programme it was 
    assumed that the years specified in the array would be contiguous however this could 
    be easily extended by modifying how the files are named
    :param years_to_scrape - all the seasons that should be scraped. The number 2015 represents
    the season starting in October 2015 and ending in July 2016
    '''

    def scrape_all_training_data(self, years_to_scrape=None):
        if years_to_scrape is None:
            years_to_scrape = [2015, 2016, 2017, 2018]
        for year in years_to_scrape:
            self.generate_game_stats(year)
        self.stitch_local_csvs(years_to_scrape)

    '''
    combines the raw data scraped over multiple seasons into a single CSV
    :param years_to_scrape all the seasons that you want in one file. This function 
    should only be used after scrapeAllTraining data has been called over the 
    same interval 
    '''

    @staticmethod
    def stitch_local_csvs(years_to_scrape=None):
        if years_to_scrape is None:
            years_to_scrape = [2015, 2016, 2017, 2018]
        output_filename = (f"data/training_data/"
                           f"training_data_{years_to_scrape[0]}-{years_to_scrape[len(years_to_scrape) - 1]}.csv")
        output_file = open(output_filename, "w")
        for index, year in enumerate(years_to_scrape):
            current_csv = pd.read_csv(f"data/game_stats/{year}-{year + 1}.csv")
            if index == 0:
                headers = current_csv.head()
                writer = csv.DictWriter(output_file, fieldnames=headers)
                writer.writeheader()
                output_file.close()

            current_csv.to_csv(f'{output_filename}', mode='a', header=False, index=False)

    '''
    combines the above scrapeAllTrainingData and stitchLocalCsv functions together
    '''

    def generate_multiple_years(self, years_to_generate=None):
        if years_to_generate is None:
            years_to_generate = [2015, 2016, 2017, 2018]
        self.stitch_local_csvs(years_to_scrape=years_to_generate)
        filepath = f"data/training_data/training_data_{years_to_generate[0]}-{years_to_generate[-1]}.csv"
        output_filename = f"data/training_features/training_features_{str(years_to_generate)[1:-1]}.csv"
        if Path(output_filename).is_file():
            os.remove(output_filename)
        games_frame = pd.read_csv(filepath)
        for year in years_to_generate:
            self._year_to_generate = year
            current_frame = games_frame.query(f"season_id == {year}")
            self.generate(data_frame=current_frame, output_location=output_filename, append=True)

        print("Finished generating the training features")
