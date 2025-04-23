from neo4j import GraphDatabase
from itertools import combinations

class SkillGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_node(self, label, name, properties=None):
        """Creates or updates a node in Neo4j."""
        with self.driver.session() as session:
            session.run(
                f"""
                MERGE (n:{label} {{name: $name}})
                SET n += $props
                """,
                name=name,
                props=properties or {}
            )
    
    def create_relationship(self, child_name, child_label, parent_name, parent_label):
        """Creates a BELONGS_TO relationship from a child node to a parent node without labels."""
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (child:{child_label} {{name: $child_name}})
                MATCH (parent:{parent_label} {{name: $parent_name}})
                MERGE (child)-[:BELONGS_TO]->(parent)
                """,
                child_name=child_name,
                parent_name=parent_name
            )
    
    def get_level_nodes(self, label="L1"):
        """
        Returns nodes (by default the originally added 'Skill' nodes)
        that do not already have an incoming BELONGS_TO relationship.
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n:{label})
                RETURN n.name as name, labels(n) AS label, n.level as level
                """
            )
            return list(result.data())
    
    def get_top_level_nodes(self):
        """
        Returns all nodes (regardless of label) that do not have any incoming
        BELONGS_TO relationships. This will include individual skills that have not
        been grouped and grouping nodes (SkillCategory) that are not assigned to a higher group.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE NOT (n)-[:BELONGS_TO]->()
                RETURN n.name AS name, labels(n) AS label, n.level AS level
                """
            )
            return list(result.data())
    
    def change_node_label(self, name, old_label, new_label):
        """Change a node's label in Neo4j by removing the old one and adding the new one."""
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (n:{old_label} {{name: $name}})
                REMOVE n:{old_label}
                SET n:{new_label}
                """,
                name=name
            )
            
    def get_L2_to_L1_skills(self):
        """
        Returns a mapping from L2 node names to the set of L1 skill names that belong to them.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (child:L1)-[:BELONGS_TO]->(parent:L2)
                RETURN parent.name AS l2_name, collect(child.name) AS skills
                """
            )
            return {record["l2_name"]: set(record["skills"]) for record in result}

    def find_subcategory_L2_pairs(self):
        """
        Finds pairs of L2 nodes where one is a strict subcategory of the other based on skill sets.
        Returns a list of tuples (subset_header, superset_header)
        """
        l2_to_skills = self.get_L2_to_L1_skills()
        subcategory_pairs = []

        for a, b in combinations(l2_to_skills.items(), 2):
            name_a, skills_a = a
            name_b, skills_b = b
            if skills_a != skills_b:
                if skills_a.issubset(skills_b):
                    subcategory_pairs.append((name_a, name_b))  # A is subset of B
                elif skills_b.issubset(skills_a):
                    subcategory_pairs.append((name_b, name_a))  # B is subset of A

        return subcategory_pairs            