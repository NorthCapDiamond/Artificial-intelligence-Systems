from user_io import checked_input
from my_lib import *
from rdflib import Graph, Literal
from requests import show_the_racers_from_team

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

	#Type of question1 : "show racers from team param"
	split_input = user_input.split()
	if split_input[0] == "show":
		if(len(split_input)>4):
			team_name = merge_underlined(split_input[4:])
			print("Team:", team_name)
			answer = show_the_racers_from_team(team_name, g)
			print("Racers:", ", ".join(answer) if len(answer) > 0 else "None")
		else:
			print("Incorrect Input")



	user_input = checked_input("Ask me something? Or type 'exit' to finish this session")