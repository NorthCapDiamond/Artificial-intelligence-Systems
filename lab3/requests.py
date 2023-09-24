from rdflib import Graph, Literal


def show_the_racers_from_team(team_name, g):
    qres = g.query(
        f"""
        SELECT (SUBSTR(str(?racer), STRLEN(str(<http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#>))+1) AS ?name)
        WHERE {{
            ?racer rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racer> .
            ?racer <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Be_In_Team> ?team .
            ?team rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racing_Team> .
            FILTER(?team = <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#{team_name}>)
        }}
        ORDER BY ?name
        """)

    answer = []

    # Выводим результаты
    for row in qres:
        answer.append(str(row[0]))

    return answer


def info_param(parameter, g):
    qres = g.query(
        f"""
        SELECT (SUBSTR(str(?racer), STRLEN(str(<http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#>))+1) AS ?name)
        WHERE {{
            ?racer rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#{parameter}> .
        }}
        """)

    answer = []

    # Выводим результаты
    for row in qres:
        answer.append(str(row[0]))

    return answer

def show_racers(team_name, g, love):
    answer = []
    filter_operator = "=" if love else "!="

    qres = g.query(
        f"""
        SELECT (SUBSTR(str(?racer), STRLEN(str(<http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#>))+1) AS ?name)
        WHERE {{
            ?racer rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racer> .
            ?racer <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Be_In_Team> ?team .
            ?team rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racing_Team> .
            FILTER(?team {filter_operator} <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#{team_name}>)
        }}
        ORDER BY ?name
        """)

    # Выводим результаты
    for row in qres:
        answer.append(str(row[0]))

    return answer


def show_winner_of_race(race, g, love):
    answer = []
    filter_operator = "=" if love else "!="
    qres = g.query(
        f"""
        SELECT (SUBSTR(str(?racer), STRLEN(str(<http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#>))+1) AS ?name)
        WHERE {{
            ?racer rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racer> .
        }}
        ORDER BY ?name
        """)


    for row in qres:
        answer.append(str(row[0]))

    qres = g.query(
        f"""
        SELECT (SUBSTR(str(?racer), STRLEN(str(<http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#>))+1) AS ?name)
        WHERE {{
            ?racer rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racer> .
            ?racer <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Win_The_Race> ?race .
            ?race rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Race> .
            FILTER(?race = <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#{race}>)
        }}
        ORDER BY ?name
        """)

    answer2 = []
    for row in qres:
        answer2.append(str(row[0]))
        answer.remove(str(row[0]))

    if(filter_operator=="="):
        return answer2
    else:
        return answer


def racers_dnf(g, love):

    answer = []
    answer2 = []
    aanswer2  = []
    filter_operator = "=" if love else "!="
    qres = g.query(
        f"""
        SELECT (SUBSTR(str(?racer), STRLEN(str(<http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#>))+1) AS ?name)
        WHERE {{
            ?racer rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racer> .
        }}
        ORDER BY ?name
        """)
    for row in qres:
        answer.append(str(row[0]))

    qres = g.query(
        f"""
        SELECT ?racer ?race
        WHERE {{
            ?racer rdf:type/rdfs:subClassOf* <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racer> .
            ?racer <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#DNF> ?race .
        }}
        """)

    for row in qres:
            answer2.append(str(row[0]).split('#')[1] + " In " + str(row[1]).split('#')[1])
            aanswer2.append([str(row[0]).split('#')[1],  str(row[1]).split('#')[1]])

    if(filter_operator=="="):
        return answer2
    else:
        for i in aanswer2:
            if(i[0] in answer):
                answer.remove(i[0])
    return answer



def teams_dnf(g, love):

    answer = []
    answer2 = []
    filter_operator = "=" if love else "!="
    qres = g.query(
        f"""
        SELECT (SUBSTR(str(?team), STRLEN(str(<http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#>))+1) AS ?name)
        WHERE {{
            ?team rdf:type <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racing_Team> .
        }}
        ORDER BY ?name
        """)
    for row in qres:
        answer.append(str(row[0]))

    qres = g.query(
        f"""
        SELECT ?team ?race
        WHERE {{
            ?racer rdf:type/rdfs:subClassOf* <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Racer> .
            ?racer <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#DNF> ?race .
            ?racer <http://www.semanticweb.org/dmitrydrobysh/ontologies/2023/8/untitled-ontology-4#Be_In_Team> ?team .
        }}
        """)

    for row in qres:
        answer2.append(str(row[0]).split('#')[1] + " In " + str(row[1]).split('#')[1])

    if(filter_operator=="="):
        return answer2
    else:
        for i in answer2:
            if(i[0] in answer):
                answer.remove(i[0])
        return answer

