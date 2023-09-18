:- discontiguous
	racer/1, race/1, race_winner/2, race_second/2, race_third/2, 
	racing_team/1, pair_of_racers/2, event/1, racer_in_team/2, 
	event_race/2, not_finished/2.


% rules 

is_winner_of_race(Race, Racer):-
	race_winner(Race, Racer).

is_second_in_race(Race, Racer):-
	race_second(Race, Racer).

is_third_in_race(Race, Racer):-
	race_third(Race, Racer).	


is_race_winner_in_team(Race, Team):-
	racer_in_team(Racer, Team), race_winner(Race, Racer).

is_race_second_in_team(Race, Team):-
	racer_in_team(Racer, Team), race_second(Race, Racer).

is_race_third_in_team(Race, Team):-
	racer_in_team(Racer, Team), race_third(Race, Racer).


is_team_on_podium(Race, Team):-
	is_race_winner_in_team(Race, Team); 
	is_race_second_in_team(Race, Team); 
	is_race_third_in_team(Race, Team).



% facts

% Here I have racers and their teams
racing_team("Red Bull").

racer("Max Verstappen").
racer_in_team("Max Verstappen", "Red Bull").

racer("Sergio Pérez").
racer_in_team("Sergio Pérez", "Red Bull").

pair_of_racers("Max Verstappen", "Sergio Pérez").




racing_team("Ferrari").

racer("Charles Leclerc").
racer_in_team("Charles Leclerc", "Ferrari").

racer("Carlos Sainz Jr.").
racer_in_team("Carlos Sainz Jr.", "Ferrari").

pair_of_racers("Charles Leclerc", "Carlos Sainz Jr.").




racing_team("Mercedes").

racer("George Russell").
racer_in_team("George Russell", "Mercedes").

racer("Lewis Hamilton").
racer_in_team("Lewis Hamilton", "Mercedes").

pair_of_racers("George Russell", "Lewis Hamilton").




racing_team("McLaren").

racer("Lando Norris").
racer_in_team("Lando Norris", "McLaren").

racer("Oscar Piastri").
racer_in_team("Oscar Piastri", "McLaren").

pair_of_racers("Lando Norris", "Oscar Piastri").




racing_team("Aston Martin").

racer("Fernando Alonso").
racer_in_team("Fernando Alonso", "Aston Martin").

racer("Lance Stroll").
racer_in_team("Lance Stroll", "Aston Martin").

pair_of_racers("Fernando Alonso", "Lance Stroll").




racing_team("Alpine").

racer("Pierre Gasly").
racer_in_team("Pierre Gasly", "Alpine").

racer("Esteban Ocon").
racer_in_team("Esteban Ocon", "Alpine").

pair_of_racers("Pierre Gasly", "Esteban Ocon").




racing_team("Alfa Romeo").

racer("Zhou Guanyu").
racer_in_team("Zhou Guanyu", "Alfa Romeo").

racer("Valtteri Bottas").
racer_in_team("Valtteri Bottas", "Alfa Romeo").

pair_of_racers("Zhou Guanyu", "Valtteri Bottas").




racing_team("AlphaTauri").

racer("Liam Lawson").
racer_in_team("Liam Lawson", "AlphaTauri").

racer("Yuki Tsunoda").
racer_in_team("Yuki Tsunoda", "AlphaTauri").

pair_of_racers("Liam Lawson", "Yuki Tsunoda").




racing_team("Williams").

racer("Alex Albon").
racer_in_team("Alex Albon", "Williams").

racer("Logan Sargeant").
racer_in_team("Logan Sargeant", "Williams").

pair_of_racers("Alex Albon", "Logan Sargeant").




racing_team("Haas").

racer("Nico Hülkenberg").
racer_in_team("Nico Hülkenberg", "Haas").

racer("Kevin Magnussen").
racer_in_team("Kevin Magnussen", "Haas").

pair_of_racers("Nico Hülkenberg", "Kevin Magnussen").

% events: 

event("Electrical failure").
event("Mechanical failure").
event("ERS failure").
event("Fire").
% Here I have races and a some useful things about them

race("BAHRAIN GP").

race_winner("BAHRAIN GP", "Max Verstappen").
race_second("BAHRAIN GP", "Sergio Pérez").
race_third("BAHRAIN GP", "Fernando Alonso").

not_finished("BAHRAIN GP", "Esteban Ocon").
not_finished("BAHRAIN GP", "Charles Leclerc").
not_finished("BAHRAIN GP", "Oscar Piastri").

event_race("BAHRAIN GP", "Electrical failure").
event_race("BAHRAIN GP", "Mechanical failure").



race("SAUDI ARABIAN GP").

race_winner("SAUDI ARABIAN GP", "Sergio Pérez").
race_second("SAUDI ARABIAN GP", "Max Verstappen").
race_third("SAUDI ARABIAN GP", "Fernando Alonso").

not_finished("SAUDI ARABIAN GP", "Alex Albon").
not_finished("SAUDI ARABIAN GP", "Lance Stroll").

event_race("SAUDI ARABIAN GP", "Mechanical failure").



race("AUSTRALIAN GP").

race_winner("AUSTRALIAN GP", "Max Verstappen").
race_second("AUSTRALIAN GP", "Lewis Hamilton").
race_third("AUSTRALIAN GP", "Fernando Alonso").

not_finished("AUSTRALIAN GP", "Alex Albon").
not_finished("AUSTRALIAN GP", "George Russell").
not_finished("AUSTRALIAN GP", "Charles Leclerc").
not_finished("AUSTRALIAN GP", "Logan Sargeant").
not_finished("AUSTRALIAN GP", "Kevin Magnussen").

event_race("AUSTRALIAN GP", "ERS failure").












