from datetime import datetime
from csv import writer

def leaderboard(agent, mean, std, file = "results/leaderboard.csv"):
    stream = open(file, 'a+')
    now = datetime.now()
    row_contents = [agent,
                    mean, 
                    std,
                    now]
    csv_writer = writer(stream)
    csv_writer.writerow(row_contents)
