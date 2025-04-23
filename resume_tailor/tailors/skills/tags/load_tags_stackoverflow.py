import xml.etree.ElementTree as ET
from resume_tailor.tailors.skills.tags.tag_utils import apply_stems_or_lemmas
import resume_tailor.tailors.skills.tags.tag_config as tcfg
import os
import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def parse_tag_counts_from_xml(xml_path):
    tag_counts = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for elem in root.findall("row"):
        tag = elem.attrib.get("TagName")
        count = int(elem.attrib.get("Count", 0))
        tag_counts[tag] = count

    return tag_counts

def parse_stackoverflow_tags():
    """
    Parses StackOverflow tags and returns a dict {tag_name: count}.
    Caches intermediate and final results using pickle.
    """
    # TODO: Update it with standardized tags
    tagset_path = tcfg.dataset_folder / "parsed_tagset.pkl"
    counts_path = tcfg.dataset_folder / "parsed_tag_counts.pkl"
    xml_path = tcfg.dataset_folder / "stackoverflow_tags.xml"

    if tagset_path.exists() and counts_path.exists():
        return load_pickle(tagset_path), load_pickle(counts_path)
    else:
        tag_counts = parse_tag_counts_from_xml(xml_path)
        save_pickle(tag_counts, counts_path)

    tag_set = lemmatize_tags(tag_counts)
    save_pickle(tag_set, tagset_path)
    
    return tag_set, tag_counts


def lemmatize_tags(tag_counts):
    tag_set = set(tag_counts.keys())
    tag_set |= set(apply_stems_or_lemmas(list(tag_set), _stemming=False))
    return tag_set

if __name__ == "__main__":
    tag_counts = parse_stackoverflow_tags()