import json
from datetime import timedelta, datetime, date
from typing import Union

import pandas as pd
import requests
from lxml import html

from Team.TeamStats import TeamStats

URL_TEMPLATE = "https://www.basketball-reference.com/boxscores/?month={month}&day={day}&year={year}"


class ScoreScraper:
    def __init__(self, start_date: Union[datetime, date], end_date: Union[datetime, date] = None) -> None:
        """

        :param start_date: start date to begin scraping data
        :param end_date: end date to begin scraping data
        """
        self._start_date = start_date
        self._end_date = start_date if end_date is None else end_date  # if no end date, just get the one start date
        self.results_list = []

        self.teams_frame = pd.read_csv("data/teams.csv")[['TEAM_ID', 'ABBREVIATION']]  # get team ids and abbreviations
        elo_frame = pd.read_csv("data/nba_elo.csv")  # read the elo data

        date_list = pd.date_range(start=start_date, end=end_date).to_list()  # make list of dates
        date_list = [str(current_date.date()) for current_date in date_list]
        dates = pd.DataFrame(date_list, columns=["date"])
        self.elo_frame = elo_frame[elo_frame['date'].isin(dates["date"])]  # get elo data for given dates

        with open("Team/team_config.json") as team_config:
            self._team_name_to_id_dict = json.load(team_config)
            teams = self._team_name_to_id_dict.values()  # make dict of team names to team ids
        team_stats = TeamStats(teams, start_date.year)  # get teams stats for that season
        for scrape_date in ScoreScraper.date_range(self._start_date, self._end_date):  # iterate over dates
            print(f"Fetching data for: {scrape_date}")
            current_date_url = URL_TEMPLATE.format(month=scrape_date.month, day=scrape_date.day, year=scrape_date.year)
            # populate url template with correct values
            response_data = requests.get(current_date_url)  # http get request of webpage

            tree = html.fromstring(response_data.content)
            games_results_list = tree.xpath("//table[@class='teams']")  # get the winning and losing teams for that date

            for game_result_element in games_results_list:
                # build up dictionary of data for each game on that date (who won, etc)
                game_results_dict = {'date': str(scrape_date), 'season_id': self._start_date.year}
                game_results_dict = self.get_teams_and_scores_dict(game_result_element, game_results_dict)
                game_result = self.get_game_result(game_result_element)

                home_team_id = game_results_dict["home_team_id"]
                away_team_id = game_results_dict["away_team_id"]
                home_team = team_stats.get_team(home_team_id)
                away_team = team_stats.get_team(away_team_id)

                self.get_home_and_road_record(game_results_dict, home_team, away_team)

                # add the elo to the game stats
                game_results_dict.update(self.get_elo(home_team_id, away_team_id, str(scrape_date)))

                # add the head to head stats to the game stats
                home_team_hth_record, away_team_hth_record = team_stats.get_head_to_head_data(home_team_id,
                                                                                              away_team_id)
                game_results_dict.update(
                    {"home_team_hth_record": home_team_hth_record, "away_team_hth_record": away_team_hth_record})

                self.get_win_loss_stats(game_results_dict, home_team, away_team)
                game_results_dict["is_home_winner"] = game_result

                # record the game to compute stats on the fly
                team_stats.record_game(
                    {"HOME_TEAM": game_results_dict["home_team_id"], "AWAY_TEAM": game_results_dict["away_team_id"],
                     "RESULT": game_results_dict["is_home_winner"],
                     "HOME_TEAM_POINTS": game_results_dict['home_team_score'],
                     "AWAY_TEAM_POINTS": game_results_dict['away_team_score']})
                self.results_list.append(game_results_dict)

    @staticmethod
    def date_range(start_date, end_date):
        """
        acts as the built-in range function but for dates
        :param start_date: date to start the "loop"
        :param end_date: end date to end the "loop"
        :return: current date by yield
        """
        for day_offset in range((end_date - start_date + timedelta(days=1)).days):
            yield start_date + timedelta(day_offset)

    @staticmethod
    def get_home_and_road_record(game_results_dict, home_team, away_team):
        """
        Populate passed in dict with home & road wins & loses
        :param game_results_dict: the dict to build up containing the necessary info
        :param home_team: instance of team representing the home team
        :param away_team: instance of team representing the away team
        :return: None
        """
        home_team_record = home_team.get_team_record()
        away_team_record = away_team.get_team_record()
        game_results_dict["home_team_home_wins"] = home_team_record["HOME_WINS"]
        game_results_dict["home_team_home_loses"] = home_team_record["HOME_LOSES"]
        game_results_dict["home_team_road_wins"] = home_team_record["AWAY_WINS"]
        game_results_dict["home_team_road_loses"] = home_team_record["AWAY_LOSES"]
        game_results_dict["away_team_home_wins"] = away_team_record["HOME_WINS"]
        game_results_dict["away_team_home_loses"] = away_team_record["HOME_LOSES"]
        game_results_dict["away_team_road_wins"] = away_team_record["AWAY_WINS"]
        game_results_dict["away_team_road_loses"] = away_team_record["AWAY_LOSES"]

    @staticmethod
    def get_win_loss_stats(team_score_dict, home_team, away_team):
        """
        Populate passed in dict with home & road wins & loss statistics
        :param team_score_dict: the dict to build up containing the necessary info
        :param home_team: instance of team representing the home team
        :param away_team: instance of team representing the away team
        :return: None
        """
        team_score_dict["home_team_wins"] = home_team.get_wins()
        team_score_dict["home_team_loses"] = home_team.get_loses()
        team_score_dict["home_team_points_per_game"] = home_team.get_points_per_game()
        team_score_dict["home_team_points_against_per_game"] = home_team.get_points_conceded_per_game()
        team_score_dict["away_team_wins"] = away_team.get_wins()
        team_score_dict["away_team_loses"] = away_team.get_loses()
        team_score_dict["away_team_points_per_game"] = away_team.get_points_per_game()
        team_score_dict["away_team_points_against_per_game"] = away_team.get_points_conceded_per_game()

    def get_teams_and_scores_dict(self, game_result_element, team_score_dict):
        """
        parses the passed in html element using Xpath to get the winning team, losing team, scores, etc
        :param game_result_element: html element containing info
        :param team_score_dict: dict to add information to
        :return: team_score_dict with additional info
        """
        winner_loser_order = game_result_element.xpath(".//tr[@class='winner' or @class='loser']")
        is_home_winner = winner_loser_order[1].attrib['class'] == 'winner'

        winning_team = str(game_result_element.xpath(f".//tr[@class='winner']/td/a/text()")[0])
        losing_team = str(game_result_element.xpath(f".//tr[@class='loser']/td/a/text()")[0])
        winning_score = int(game_result_element.xpath(f".//tr[@class='winner']/td[@class='right']/text()")[0])
        losing_score = int(game_result_element.xpath(f".//tr[@class='loser']/td[@class='right']/text()")[0])

        home_team = winning_team if is_home_winner else losing_team
        away_team = losing_team if is_home_winner else winning_team

        team_score_dict['home_team'] = home_team
        team_score_dict['home_team_id'] = self._team_name_to_id_dict[home_team]
        team_score_dict['home_team_score'] = winning_score if is_home_winner else losing_score
        team_score_dict['away_team'] = away_team
        team_score_dict['away_team_id'] = self._team_name_to_id_dict[away_team]
        team_score_dict['away_team_score'] = losing_score if is_home_winner else winning_score
        return team_score_dict

    @staticmethod
    def get_game_result(game_result_element):
        """
        :param game_result_element: html element containing data
        :return: 1 if home team won game, else 0
        """
        winner_loser_order = game_result_element.xpath(".//tr[@class='winner' or @class='loser']")
        is_home_winner = winner_loser_order[1].attrib['class'] == 'winner'
        return 1 if is_home_winner else 0

    def get_elo(self, home_team_id, away_team_id, game_date):
        """
        Gets the elo & raptors stats for both team ids from self.elo_frame from __init__
        :param home_team_id:
        :param away_team_id:
        :param game_date: date to get the relevant statistics
        :return: dict containing home & away team elo and raptor stats
        """
        elo_dict = {}

        home_team_abbreviation = self.teams_frame[self.teams_frame['TEAM_ID'] == home_team_id]['ABBREVIATION'].iloc[0]
        away_team_abbreviation = self.teams_frame[self.teams_frame['TEAM_ID'] == away_team_id]['ABBREVIATION'].iloc[0]

        current_game_elo = self.elo_frame[self.elo_frame['date'] == game_date]
        current_game_elo = current_game_elo[current_game_elo['team1'] == home_team_abbreviation]
        current_game_elo = current_game_elo[current_game_elo['team2'] == away_team_abbreviation]

        elo_dict["home_team_elo"] = current_game_elo['elo1_pre'].iloc[0]
        elo_dict["away_team_elo"] = current_game_elo['elo2_pre'].iloc[0]
        elo_dict["home_team_raptor"] = current_game_elo['raptor1_pre'].iloc[0]
        elo_dict["away_team_raptor"] = current_game_elo['raptor2_pre'].iloc[0]

        return elo_dict
