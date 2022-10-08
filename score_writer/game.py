from collections import OrderedDict

NUMBER_OF_GAMES = 3


class Game(OrderedDict):
    """
    Represents a game played. This class is used to write the relevant feature data to the csv.
    Instances of this game are an ordered dictionary (a regular python dictionary that keeps track of the orders
    key value pairs were added. Needed for adding features to the csv. Also why HOME_TEAM_WINS must be last, as it is
    the output/label value. __setitem__ adds key value pairs to the dict. These keys denote features.
    """

    # TODO remove unused params
    def __init__(self, home_team, away_team, home_team_win, home_team_elo, away_team_elo,
                 home_team_raptor, away_team_raptor, home_team_hth, away_team_hth):
        """
        Adds the features (keys) to the OrderedDict along with their values

        :param home_team: instance of Team representing home team
        :param away_team:instance of Team representing away team
        :param home_team_win: boolean flag 0/1 denoting if the home team has won
        :param home_team_elo:
        :param away_team_elo:
        :param home_team_raptor:
        :param away_team_raptor:
        :param home_team_hth:
        :param away_team_hth:
        """
        super().__init__()

        home_team_record = home_team.get_team_record()
        away_team_record = away_team.get_team_record()
        super().__setitem__("HOME_TEAM_HOME_WINS", home_team_record["HOME_WINS"])
        # super().__setitem__("HOME_TEAM_HOME_LOSES", home_team_record["HOME_LOSES"])
        super().__setitem__("HOME_TEAM_ROAD_WINS", home_team_record["AWAY_WINS"])
        super().__setitem__("HOME_TEAM_ROAD_LOSES", home_team_record["AWAY_LOSES"])

        # super().__setitem__("AWAY_TEAM_HOME_WINS", away_team_record["HOME_WINS"])
        # super().__setitem__("AWAY_TEAM_HOME_LOSES", away_team_record["HOME_LOSES"])
        super().__setitem__("AWAY_TEAM_ROAD_WINS", away_team_record["AWAY_WINS"])
        super().__setitem__("AWAY_TEAM_ROAD_LOSES", away_team_record["AWAY_LOSES"])

        super().__setitem__("HOME_TEAM_HTH_RECORD", home_team_hth)
        # super().__setitem__("AWAY_TEAM_HTH_RECORD", away_team_hth)

        '''
        get the teams record of the last 3 games
        '''
        home_team_history = home_team.get_current_form()
        away_team_history = away_team.get_current_form()

        for game in range(0,NUMBER_OF_GAMES):
            if game < 2:
                super().__setitem__(f"AWAY_TEAM_FORM_{game}", away_team_history[game])
            else:
                super().__setitem__(f"HOME_TEAM_FORM_{game}", home_team_history[game])


        super().__setitem__("HOME_TEAM_WIN_RECORD", home_team.get_wins())
        super().__setitem__("AWAY_TEAM_WIN_RECORD", away_team.get_wins())

        super().__setitem__("HOME_TEAM_PPG", home_team.get_points_per_game())
        super().__setitem__("HOME_TEAM_PAPG", home_team.get_points_conceded_per_game())
        super().__setitem__("AWAY_TEAM_PPG", away_team.get_points_per_game())
        super().__setitem__("AWAY_TEAM_PAPG", away_team.get_points_conceded_per_game())

        '''
        add the elo ratings for each game from https://projects.fivethirtyeight.com/nba-model/nba_elo.csv
        '''
        # super().__setitem__("HOME_TEAM_ELO", home_team_elo)
        # super().__setitem__("AWAY_TEAM_ELO", away_team_elo)

        '''
        MAKE SURE THIS IS ADDED LAST
        '''
        super().__setitem__("HOME_TEAM_WINS", home_team_win)
