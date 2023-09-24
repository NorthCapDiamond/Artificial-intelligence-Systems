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
