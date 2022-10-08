import csv
import json
import os
from pathlib import Path

import requests
from lxml import html

'''
We had an idea that Team A's performance in the last year against Team B 
might give us an indication as to which team will win this time. E.g if Team A
won the last 4 games that they played against Team B then it is likely that they 
will win the next game that they play against TeamB. With this assumption we wrote 
this scraper to extract this data from basketballreference.com
'''

# The page we were trying to scrape was rendered using Javascript so it was necessary to proxy the request through a
# node server to render the JS and return the result.

PUPPETEER_URL_PREFIX = "http://localhost:3000?url="
URL_TEMPLATE = PUPPETEER_URL_PREFIX + ("https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
                                       "#expanded_standings::none")


def get_team_id_from_abbreviation(team_name_to_id_dict, team_abbreviation):
    """
    :param team_name_to_id_dict: dictionary of team abbr to id
    :param team_abbreviation: The Abbreviation of a team (CHI)
    :return: The ID associated with the team per team_config.json
    """
    if team_abbreviation == "Team":
        return team_abbreviation
    return team_name_to_id_dict[team_abbreviation]


def get_team_id(team_name_to_id_dict, team_name):
    """
    :param team_name_to_id_dict: dictionary of team abbr to id
    :param team_name: the English name of the team e.g Chicago Bulls
    :return: The ID associated with the team per team_config.json
    """
    name_array = team_name.split(" ")

    if name_array[0] == "Los":
        franchise = f"LA {name_array[2]}"
    elif len(name_array) > 2 and name_array[0] != "Portland":
        franchise = f"{name_array[0]} {name_array[1]}"
    else:
        franchise = name_array[0]
    return team_name_to_id_dict[franchise]


def main():
    with open("../Team/team_config.json") as team_config:
        team_name_to_id_dict = json.load(team_config)

    start_year = int(input("Choose a year to get the head to head data for\n > "))
    output_file = f"../data/head_to_head/{start_year}.csv"

    is_file_existing = Path(output_file).is_file()

    if is_file_existing:
        os.remove(output_file)

    with open(output_file, 'a') as output_csv:
        # fetch the data about a particular year
        current_date_url = URL_TEMPLATE.format(year=start_year)
        response_data = requests.get(current_date_url)
        tree = html.fromstring(response_data.content)

        # using XPaths extract table headings and rows from the table we want
        head_to_head_table = tree.xpath('//*[@id="team_vs_team"]')
        table_rows = head_to_head_table[0].xpath('.//tr')
        table_headings = table_rows[0].xpath('.//th')[1:]

        # Taking the headings from the table and add them to a dict. This is all the team names abbreviated
        # e.g CHI - Chicago Bulls
        column_headings = [heading.text for heading in table_headings]
        encoded_column_headings = []
        for heading in column_headings:
            encoded_heading = get_team_id_from_abbreviation(team_name_to_id_dict, heading)
            encoded_column_headings.append(encoded_heading)
        writer = csv.DictWriter(output_csv, fieldnames=encoded_column_headings, lineterminator='\n')
        writer.writeheader()

        # iterate over all the rows in the table and extract the head to head matrix that we want e.g
        '''
           A  B
        A  -  0-2
        B  2-0 -
        '''
        table_rows = table_rows[1:len(table_rows)]
        for row in table_rows:
            rank_cell = row.xpath(f'th[@data-stat="ranker"]')
            row_has_data = len(rank_cell) > 0
            if row_has_data and rank_cell[0].text != "Rk":
                data_cells = row.xpath("td")
                team_cell = data_cells[0]
                team_name = team_cell.xpath("a")[0].text
                output_dict = {"Team": get_team_id(team_name_to_id_dict, team_name)}
                for index, data_cells in enumerate(data_cells):
                    if index > 0:
                        output_dict[encoded_column_headings[index]] = data_cells.text
                writer.writerow(output_dict)


if __name__ == '__main__':
    main()
