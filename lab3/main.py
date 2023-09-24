from user_io import checked_input
from my_lib import *
from rdflib import Graph, Literal
from requests import *

# создаем граф из файла
g = Graph()
g.parse("lab2.rdf", format="xml")



user_name = checked_input("Hello, It's Alexey Popov, Your Formula 1 Assistant! What is your name?")
while (not(len(user_name))):
	user_name = checked_input("You need to enter your name")

print("Nice to meet you,", user_name + ",", "What do you want know about Formula 1")


user_input = checked_input("Ask me something? Or type 'exit' to finish this session")
while(user_input.strip()!="exit"):
	while(not(len(user_input))):
		user_input = checked_input("Incorrect input")

	user_input = user_input.strip()
	split_input = user_input.split(" ")

	#Type of question0 : info param

	if(split_input[0] == "info"):
		if(len(split_input)>1):
			parameter = merge_underlined(split_input[1:])
			info_param_answer = info_param(parameter, g)
			string_bad =  "You can ask for 'Race', 'Racer', 'Event', 'Racing Team'"
			print("info:", "\n".join(info_param_answer) if len(info_param_answer) > 0 else string_bad)
		else:
			print("Incorrect Input")


	#Type of question1 : "show racers from team param"
	if split_input[0] == "show":
		if(len(split_input)>4):
			team_name = merge_underlined(split_input[4:])
			print("Team:", team_name)
			answer = show_the_racers_from_team(team_name, g)
			print("Racers:", ", ".join(answer) if len(answer) > 0 else "None")
		else:
			print("Incorrect Input")


	#Type of question2 : // i like/hate to see the team that dnf  
	if(len(split_input) > 7):
		if(split_input[-1] == "dnf" and split_input[5] == "team"):
			love = True if split_input[1] == "like" else False
			answer = teams_dnf(g, love)
			print("Teams:")
			print("\n".join(answer) if len(answer) > 0 else "None")


	#Type of question3 : // i like/hate to see the racer that dnf  
	if(len(split_input) > 7):
		if(split_input[-1] == "dnf" and split_input[5] == "racer"):
			love = True if split_input[1] == "like" else False
			answer = racers_dnf(g, love)
			print("Racers:")
			print("\n".join(answer) if len(answer) > 0 else "None")


	#Type of question4 : // i like/hate to see the winner of param
	if(len(split_input) > 7):
		if(split_input[5] == "winner"):
			race_name = merge_underlined(split_input[7:])
			love = True if split_input[1] == "like" else False
			answer = show_winner_of_race(race_name, g, love)
			print("Racers:")
			print("\n".join(answer) if len(answer) > 0 else "None")

	#Type of question5 :  i like/hate param racers
	if(len(split_input) > 3):
		if(split_input[-1] == "racers"):
			love = True if split_input[1] == "like" else False
			team_name = merge_underlined(split_input[2:-1])
			answer = show_racers(team_name, g, love)
			print("Racers:")
			print("\n".join(answer) if len(answer) > 0 else "None")
	else:
		print("No such command... try again please")

	




	user_input = checked_input("Ask me something? Or type 'exit' to finish this session")


print("Have a nice day!)")
