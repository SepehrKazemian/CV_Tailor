from importlib import reload
from resume_tailor.tailors.skill_graph import skill_graph_llms as sgl
import resume_tailor.tailors.skill_graph.skill_graph_config as sgc

reload(sgl)

known_domains = [
    "Software Engineering", "Data Science", "Machine Learning", "Deep Learning", "LLM", 
    "NLP", "Computer Vision", "Cloud Computing", "DevOps", "MLOps", "Data Engineering",
    "Cybersecurity", "Web Development", "Mobile Development", "Database",
    "Business Intelligence", "Project Management", "Computer Science", "Information Retrieval"
]

class SkillDomainSubdomainClassifier:
    """
    Classifier for mapping skills -> domains -> subdomains using LLMs.
    """

    def __init__(self, raw_client, known_domains=known_domains, batch_size=20):
        """
        Initialize classifier.

        Args:
            raw_client: LLM client or service needed by skill_domain_classifier.
            known_domains (list or set): Known allowed domains. If None, uses default list.
            batch_size (int): Number of skills to process per batch.
        """
        self.raw_client = raw_client
        self.batch_size = batch_size

        if known_domains is None:
            self.known_domains = {
                "Software Engineering", "Data Science", "Machine Learning", "Deep Learning", "LLM",
                "NLP", "Computer Vision", "Cloud Computing", "DevOps", "MLOps", "Data Engineering",
                "Cybersecurity", "Web Development", "Mobile Development", "Database",
                "Business Intelligence", "Project Management", "Computer Science", "Information Retrieval"
            }
        else:
            self.known_domains = set(known_domains)

    def classify_skills_to_domains(self, standardized_skills: set) -> tuple[dict, set]:
        """
        Classify skills into domains in batches.

        Args:
            standardized_skills (set): Set of cleaned/standardized skill strings.

        Returns:
            domain_to_skills (dict): {domain: [skills]}
            new_domains (set): Domains that were not originally in known_domains.
        """
        domain_to_skills = {}
        new_domains = set()

        skills_list = list(standardized_skills)

        for i in range(0, len(skills_list), self.batch_size):
            batch = skills_list[i:i+self.batch_size]

            skill_to_domains = sgl.skill_domain_classifier(batch, self.known_domains, self.raw_client)

            for skill, domains in skill_to_domains.items():
                for domain in domains:
                    if domain not in self.known_domains:
                        new_domains.add(domain)
                    domain_to_skills.setdefault(domain, []).append(skill)

        return domain_to_skills, new_domains

    def classify_domains_to_subdomains(self, domain_to_skills: dict) -> dict:
        """
        Classify domains into subdomains using skill lists.

        Args:
            domain_to_skills (dict): {domain: [skills]}.

        Returns:
            subdomain_to_domain (dict): {subdomain: parent_domain}.
        """
        reload(sgl)
        subdomain_to_domain = sgl.classify_subdomains(domain_to_skills, self.raw_client)
        return subdomain_to_domain


    def classify_skills_to_subdomains(self, domain_to_skills: dict, domain_to_subdomain: dict) -> dict:
        """
        Classify skills into subdomains under each domain.

        Args:
            domain_to_skills (dict): {domain: [skills]}
            domain_to_subdomain (dict): {domain: [subdomains]}

        Returns:
            domain_to_subdomain_skills (dict): {domain: {subdomain: [skills]}}
        """
        reload(sgl)
        domain_to_subdomain_skills = {}

        for domain, subdomains in domain_to_subdomain.items():
            domain_skills = domain_to_skills.get(domain, [])
            if not domain_skills:
                continue  # No skills for this domain

            # Classify skills into subdomains
            subdomain_to_skills = sgl.subdomain_skill_mapping(domain, domain_skills, subdomains, self.raw_client)

            domain_to_subdomain_skills[domain] = subdomain_to_skills

        return domain_to_subdomain_skills