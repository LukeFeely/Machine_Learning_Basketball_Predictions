# NBA Machine Learning Model 🏀⛹

<p align="middle">
<img width=400 src="https://dl.dropboxusercontent.com/s/yjh9y7b60kdiwhs/bball.png?dl=0" />
</p>

## TLDR: 
- We managed to predict the games in the seasons 2015-2019 with 65% accuracy 
- Using a logisitic regression model, kNN classifier and Kernelized Support Vector Machine
- Features we used in the model
  - Average Points per game
  - Average Points conceded per game
  - Wins so far this season 
  - Head to head record from the previous season 
  - Current form of the team
  

## How to run the repo 
1. run `pip install -r requirements.txt`
2. run main.py
3. (Optionally) scrape the data respond yes to the first prompt
4. (Optionally)If you would like to change the features used in the model you can comment out some of the features added to the dictionary in game.py. After you do this you must regenerate the CSVs used to train the model so respond 'y' to the second prompt in main.py. Else respond no. 
5. To enable or disable the cross validation simply flip the flag should_run_cross_validation
