import pandas as pd

from .Team import Team

'''
Wrapper class for the Team class
Keeps a list of Team objects, record game takes in a game 
and updates the teams who played the game appropriately 
'''


class TeamStats:
    def __init__(self, team_list, current_season):
        team_map = {}
        for team_id in team_list:
            team_map[team_id] = Team(team_id)
        self.team_map = team_map
        previous_season = current_season - 1
        file_path = f"./data/head_to_head/{previous_season}.csv"
        self.head_to_head_frame = pd.read_csv(file_path)
        self.count = 0

    '''
    Takes in a game dictionary and updates the Home Team and Away team so that they can 
    keep track of stats about the team as the season progressed
    @:home_away_result_dict -> 
    {
    "HOME_TEAM" : INTEGER (this is the ID of the home team)
    "HOME_TEAM_POINTS" : INTEGER (number of points the home team scored in the game)
    "AWAY_TEAM" : INTEGER (this is the ID of the away team)
    "AWAY_TEAM_POINTS" : INTEGER (number of points that the away team scored in the game)
    "RESULT" : 1 if the home team wins else 0
    }
    '''

    def record_game(self, home_away_result_dict):
        home_team = self.team_map[home_away_result_dict["HOME_TEAM"]]
        home_team.parse_game(home_away_result_dict)

        away_team = self.team_map[home_away_result_dict["AWAY_TEAM"]]
        away_team.parse_game(home_away_result_dict)

    '''
    gets the results for the last three games for a particular team 
    @:team_id unique identifier for a team as in the teams_conf.json
    '''

    def get_team_record(self, team_id):
        current_team = self.team_map[team_id]
        return current_team.get_current_form()

    '''
    gets a Team object from the dict of teams maintained by this class
    based on that Team's team_id which are derived from the teams_conf.json
    '''

    def get_team(self, team_id):
        return self.team_map[team_id]

    '''
    gets the win loss record for the two teams during the previous season 
    params: id of home team, id of away team (as in teams.conf.json)
    returns: record e.g [3,1] means the home team won 3 times and lost once 
    during the previous season
    '''

    def get_head_to_head_data(self, home_team_id, away_team_id):
        self.count += 1
        home_team_record = self.head_to_head_frame.query(f"Team == {str(home_team_id)}")
        head_to_head_string = home_team_record[str(away_team_id)].iloc[0]
        head_to_head_list = head_to_head_string.split("-")
        return [int(i) for i in head_to_head_list]
