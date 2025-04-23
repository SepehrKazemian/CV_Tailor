from resume_tailor.tailors.skills.kg_class import SkillGraph
from resume_tailor.tailors.skills import kg_grouping_llm as kgg_llm
from collections import Counter

def initialize_skills(graph, skills):
    """
    For each tuple in skills, create a node with label 'Skill'
    and a property "level" that stores the skill level.
    """
    for skill, level in skills:
        graph.create_node("L1", skill, {str(level): 1})

def add_groupings_with_lookup(graph, grouping, skill_nodes):
    """
    Create SkillCategory nodes with a Counter-based level distribution instead of summing levels.
    Each SkillCategory will have a 'level_distribution' like {3: 2, 1: 1}.
    """
    # Create a lookup dictionary from skill name to level or Counter.
    skill_lookup = {node["name"]: node["level"] for node in skill_nodes}
    label_lookup = {node["name"]: node["label"][0] for node in skill_nodes}

    for group_name, children in grouping.items():
        if not children:
            lvl = int(label_lookup[group_name][1:])
            graph.change_node_label(group_name, old_label=label_lookup[group_name], new_label=f"L{lvl + 1}")
            continue

        level_counter = Counter()

        for child in children:
            level = skill_lookup.get(child, 0)

            if isinstance(level, dict):
                # If level is already a counter-like dict, merge it
                level_counter.update(level)
            elif isinstance(level, Counter):
                level_counter.update(level)
            elif isinstance(level, int):
                level_counter[level] += 1
            else:
                raise ValueError(f"Unsupported level format for skill '{child}': {level}")

        lvl = int(label_lookup[children[0]][1:])
        flat_props = {f"{k}": v for k, v in level_counter.items()}
        print(flat_props)
        
        # Create parent node with level_distribution as property
        graph.create_node(f"L{lvl+1}", group_name, flat_props)

        for child in children:
            graph.create_relationship(child, label_lookup[child], group_name, f"L{lvl+1}")

def grouping_has_children(grouping):
    has_children = False
    for _, arr in grouping.items():
        if len(arr) > 0:
            has_children = True
            break
    return has_children

def main(raw_client, model_name, skills):
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "neo4j123"   # change with your actual password

    # Instantiate the SkillGraph
    graph = SkillGraph(uri, user, password)

    # Define the skills list
    skills = [
        ("Python", 3),
        ("Java", 2),
        ("Machine Learning", 1),
        ("Docker", 2),
        ("Kubernetes", 1),
        ("AWS", 2),
        ("CI/CD", 1)
    ]

    # building kg
    initialize_skills(graph, skills)
    
    # getting highest level nodes (parents)
    highest_nodes = graph.get_top_level_nodes()
    nodes_str = [i["name"] for i in highest_nodes]
    
    # run llm to group parent nodes
    grouping = kgg_llm.grouping_skills(raw_client, model_name, nodes_str)
    add_groupings_with_lookup(graph, grouping, highest_nodes)
    
    has_children = True
    while has_children:
        highest_nodes = graph.get_top_level_nodes()
        nodes_str = [i["name"] for i in highest_nodes]
        grouping = kgg_llm.grouping_skill_sections(raw_client, model_name, nodes_str)
        
        has_children = grouping_has_children(grouping)
        if not has_children:
            break