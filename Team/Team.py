NUMBER_OF_GAMES = 3

'''
Not all the features that we wanted to scrape were readily available on 
basketballreference.com but we were able to derive them from other data that we had collected
e.g the win percentage of a team when they are not playing the game on their home court.
We made this class to compute statistics about a Team as we read data into the programme 
game by game. For example by aggregating the number of points a team had scored in 
all games, we were able to keep a rolling average of the number of points a team had 
scored per game at any given point of the season. 
'''


class Team:
    """
    constructor just assigns a Team ID to the team so that we can infer on a high level which entity
    the Team represents e.g the LA Lakers
    """

    def __init__(self, team_id):
        self.team_id = team_id
        self.game_history = [0] * NUMBER_OF_GAMES
        self.num_home_wins = 0
        self.num_home_loses = 0
        self.num_away_wins = 0
        self.num_away_loses = 0
        self.number_games_played = 0
        self.points_per_game = 0
        self.points_conceded_per_game = 0

    '''
    Given a dictionary which represents a game results, this method will increment statistics
    about the team based on whether they won or lost, how many points they scored etc ect

    @:param -> 
    {
    "HOME_TEAM" : INTEGER (this is the ID of the home team)
    "HOME_TEAM_POINTS" : INTEGER (number of points the home team scored in the game)
    "AWAY_TEAM" : INTEGER (this is the ID of the away team)
    "AWAY_TEAM_POINTS" : INTEGER (number of points that the away team scored in the game)
    "RESULT" : 1 if the home team wins else 0
    }
    '''

    def parse_game(self, game):
        home_team = game["HOME_TEAM"]
        home_team_win = game["RESULT"]

        team_has_won = False
        is_home_team = self.team_id == home_team

        if is_home_team:
            self.points_per_game += game["HOME_TEAM_POINTS"]
            self.points_conceded_per_game += game["AWAY_TEAM_POINTS"]
            if home_team_win:
                team_has_won = True
                self.num_home_wins += 1
            else:
                self.num_home_loses += 1
        else:
            self.points_per_game += game["AWAY_TEAM_POINTS"]
            self.points_conceded_per_game += game["HOME_TEAM_POINTS"]
            if not home_team_win:
                team_has_won = True
                self.num_away_wins += 1
            else:
                self.num_away_loses += 1

        result = 1 if team_has_won else 0
        self.number_games_played += 1
        self.game_history.insert(0, result)
        if len(self.game_history) > NUMBER_OF_GAMES:
            self.game_history.pop()

    '''
    @:param team_name the name of a team in words e.g "Portland Trail Blazers"
    :returns English representation which we can use to map a team to a team ID in the team_config.json
    e.g "Portland", "LA Lakers", "Chicago" etc etc
    '''

    @staticmethod
    def get_franchise(team_name):
        name_array = team_name.split(" ")
        if name_array[0] == "Los":
            franchise = f"LA {name_array[2]}"
        elif len(name_array) > 2 and name_array[0] != "Portland":
            franchise = f"{name_array[0]} {name_array[1]}"
        else:
            franchise = name_array[0]
        return franchise

    '''
    aggregates a teams performance over their last three games using the game_history 
    which is a LIFO queue of wins losses over the last 3 games. 
    '''

    def get_current_form(self):
        wins = 0
        loses = 0

        for game in self.game_history:
            if game == 1:
                wins += 1
            else:
                loses += 1
        return self.game_history

    '''
    returns the number of wins that a team has in the season up to a certain point
    '''

    def get_wins(self):
        return self.num_away_wins + self.num_home_wins

    '''
    returns the number of losses a team has in a season up to a certain point
    '''

    def get_loses(self):
        return self.num_away_loses + self.num_home_loses

    '''
    returns the number of games that a team has played up to a certain point in the season
    '''

    def get_number_games_played(self):
        return self.get_wins() + self.get_loses()

    '''
    returns the arithmetic average of points scored per game divided by the number of games played
    '''

    def get_points_per_game(self):
        number_games_played = self.get_number_games_played()
        if number_games_played == 0:
            return 0
        else:
            return int(self.points_per_game / self.get_number_games_played())

    '''
    returns the arithmetic average of points conceded per game divided by the number of games played
    '''

    def get_points_conceded_per_game(self):
        number_games_played = self.get_number_games_played()
        return 0 if number_games_played == 0 else int(self.points_conceded_per_game / self.get_number_games_played())

    '''
    Returns the win / loss percentage of team both when they play on their home court and when they play 
    away from their home court. The offset of +1 in the computations is to avoid a divide by zero error and results 
    in the win loss percentage being 50:50 when there has been no games played yet 
    '''

    def get_team_record(self):
        return {
            "HOME_WINS": round((1 + self.num_home_wins) / (1 + self.num_home_wins + 1 + self.num_home_loses), 3),
            "HOME_LOSES": round((1 + self.num_home_loses) / (1 + self.num_home_wins + 1 + self.num_home_loses), 3),
            "AWAY_WINS": round((1 + self.num_away_wins) / (1 + self.num_away_wins + 1 + self.num_away_loses), 3),
            "AWAY_LOSES": round((1 + self.num_away_loses) / (1 + self.num_away_wins + 1 + self.num_away_loses), 3),
        }
