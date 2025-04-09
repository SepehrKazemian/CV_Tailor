"""
Job Description Generator Module

This module provides functionality to generate random job descriptions for testing.
"""

import os
import random
import importlib.util
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import config.py dynamically
config_path = Path(__file__).resolve().parents[2] / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class JobDescriptionGenerator:
    """
    Generates random job descriptions for testing.
    """
    
    # Job titles for different domains
    JOB_TITLES = {
        "software_engineering": [
            "Software Engineer", "Senior Software Engineer", "Full Stack Developer",
            "Backend Developer", "Frontend Developer", "DevOps Engineer",
            "Site Reliability Engineer", "Software Architect", "Mobile Developer",
            "Game Developer", "Embedded Systems Engineer"
        ],
        "data_science": [
            "Data Scientist", "Machine Learning Engineer", "Data Engineer",
            "Data Analyst", "Business Intelligence Analyst", "Research Scientist",
            "AI Engineer", "MLOps Engineer", "Computer Vision Engineer", "NLP Engineer"
        ],
        "cloud_computing": [
            "Cloud Engineer", "Cloud Architect", "AWS Solutions Architect",
            "GCP Engineer", "Azure Cloud Engineer", "Cloud Security Engineer",
            "Cloud DevOps Engineer", "Kubernetes Engineer", "Cloud Infrastructure Engineer"
        ],
        "cybersecurity": [
            "Security Engineer", "Security Analyst", "Penetration Tester",
            "Security Architect", "Cybersecurity Consultant", "Information Security Analyst",
            "Security Operations Engineer", "Threat Intelligence Analyst"
        ],
        "product_management": [
            "Product Manager", "Technical Product Manager", "Product Owner",
            "Program Manager", "Project Manager", "Scrum Master",
            "Agile Coach", "Product Analyst"
        ]
    }
    
    # Skills for different domains
    SKILLS = {
        "programming_languages": [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go",
            "Rust", "Swift", "Kotlin", "PHP", "Ruby", "Scala", "R"
        ],
        "frontend": [
            "React", "Angular", "Vue.js", "Next.js", "HTML", "CSS", "SASS",
            "Tailwind CSS", "Redux", "Webpack", "Babel", "jQuery"
        ],
        "backend": [
            "Node.js", "Django", "Flask", "Spring Boot", "Express.js", "FastAPI",
            "Laravel", "Ruby on Rails", "ASP.NET", "GraphQL", "REST API"
        ],
        "databases": [
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "DynamoDB", "Cassandra",
            "Redis", "Elasticsearch", "Neo4j", "SQLite", "Oracle", "SQL Server"
        ],
        "cloud": [
            "AWS", "GCP", "Azure", "Kubernetes", "Docker", "Terraform", "CloudFormation",
            "Lambda", "EC2", "S3", "RDS", "DynamoDB", "BigQuery", "Cloud Functions"
        ],
        "data_science": [
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn",
            "Pandas", "NumPy", "Data Visualization", "Statistical Analysis", "A/B Testing",
            "Natural Language Processing", "Computer Vision", "Reinforcement Learning"
        ],
        "devops": [
            "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Ansible",
            "Puppet", "Chef", "Prometheus", "Grafana", "ELK Stack", "Datadog"
        ],
        "soft_skills": [
            "Communication", "Teamwork", "Problem Solving", "Critical Thinking",
            "Time Management", "Leadership", "Adaptability", "Creativity",
            "Attention to Detail", "Project Management", "Mentoring"
        ]
    }
    
    # Company types
    COMPANY_TYPES = [
        "Startup", "Tech Giant", "Enterprise", "Consulting Firm", "Agency",
        "Financial Institution", "Healthcare Organization", "E-commerce Company",
        "SaaS Provider", "Government Agency", "Non-profit Organization"
    ]
    
    # Experience levels
    EXPERIENCE_LEVELS = [
        "Entry Level", "Junior", "Mid-Level", "Senior", "Staff", "Principal", "Lead"
    ]
    
    def __init__(self, llm_provider: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the JobDescriptionGenerator.
        
        Args:
            llm_provider: LLM provider to use ('openai', 'google', 'anthropic', or None to use config.py)
            api_key: API key for the selected LLM provider (None to use config.py or environment variable)
        """
        self.llm_provider = llm_provider or getattr(config, 'LLM', 'claude').lower()
        self.api_key = api_key or getattr(config, 'key', None)
        self.llm_version = getattr(config, 'version', None)
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the provider."""
        if self.llm_provider == "openai":
            return ChatOpenAI(
                model="gpt-4" if not self.llm_version else f"gpt-{self.llm_version}",
                temperature=0.7,  # Higher temperature for more creativity
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
            )
        elif self.llm_provider == "google":
            return ChatGoogleGenerativeAI(
                model="gemini-pro" if not self.llm_version else f"gemini-{self.llm_version}",
                temperature=0.7,
                api_key=self.api_key or os.environ.get("GOOGLE_API_KEY")
            )
        elif self.llm_provider == "claude" or self.llm_provider == "anthropic":
            # Use a standard model name that's known to work
            model_version = "claude-3-opus-20240229"
            
            return ChatAnthropic(
                model=model_version,
                temperature=0.7,
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def generate_job_description(self, domain: Optional[str] = None, 
                                skills_to_include: Optional[List[str]] = None,
                                experience_level: Optional[str] = None) -> str:
        """
        Generate a random job description.
        
        Args:
            domain: The domain for the job description (e.g., 'software_engineering', 'data_science')
            skills_to_include: List of skills to include in the job description
            experience_level: The experience level for the job
            
        Returns:
            A randomly generated job description
        """
        # Select a random domain if not provided
        if domain is None:
            domain = random.choice(list(self.JOB_TITLES.keys()))
        
        # Select a random job title from the domain
        job_title = random.choice(self.JOB_TITLES.get(domain, self.JOB_TITLES["software_engineering"]))
        
        # Select a random company type
        company_type = random.choice(self.COMPANY_TYPES)
        
        # Select a random experience level if not provided
        if experience_level is None:
            experience_level = random.choice(self.EXPERIENCE_LEVELS)
        
        # Select random skills if not provided
        if skills_to_include is None:
            skills_to_include = []
            # Add some programming languages
            skills_to_include.extend(random.sample(self.SKILLS["programming_languages"], 
                                                 k=random.randint(1, 3)))
            
            # Add some domain-specific skills
            if domain == "software_engineering":
                skills_to_include.extend(random.sample(self.SKILLS["frontend"] + self.SKILLS["backend"], 
                                                     k=random.randint(2, 4)))
            elif domain == "data_science":
                skills_to_include.extend(random.sample(self.SKILLS["data_science"], 
                                                     k=random.randint(3, 5)))
            elif domain == "cloud_computing":
                skills_to_include.extend(random.sample(self.SKILLS["cloud"], 
                                                     k=random.randint(3, 5)))
            
            # Add some databases
            skills_to_include.extend(random.sample(self.SKILLS["databases"], 
                                                 k=random.randint(1, 2)))
            
            # Add some cloud skills
            skills_to_include.extend(random.sample(self.SKILLS["cloud"], 
                                                 k=random.randint(1, 3)))
            
            # Add some soft skills
            skills_to_include.extend(random.sample(self.SKILLS["soft_skills"], 
                                                 k=random.randint(2, 4)))
        
        # Create a prompt template
        prompt_template = PromptTemplate(
            template="""
            You are a technical recruiter writing job descriptions for tech companies.
            
            Create a detailed job description for a {experience_level} {job_title} at a {company_type}.
            
            The job description should include:
            1. A brief company overview
            2. Detailed responsibilities
            3. Required skills and qualifications
            4. Preferred/nice-to-have skills
            5. Benefits and perks
            
            Make sure to include these specific skills in the requirements: {skills}
            
            The job description should be professional, detailed, and realistic.
            Format it with clear sections and bullet points where appropriate.
            """,
            input_variables=["experience_level", "job_title", "company_type", "skills"]
        )
        
        # Create the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            # Run the chain
            result = chain.run(
                experience_level=experience_level,
                job_title=job_title,
                company_type=company_type,
                skills=", ".join(skills_to_include)
            )
            
            return result.strip()
            
        except Exception as e:
            print(f"Error generating job description: {e}")
            # Return a basic job description if the chain fails
            return f"""
            Job Title: {experience_level} {job_title}
            
            Company: A {company_type}
            
            Requirements:
            - {', '.join(skills_to_include)}
            
            Responsibilities:
            - Design, develop, and maintain software applications
            - Collaborate with cross-functional teams
            - Write clean, efficient, and maintainable code
            
            Benefits:
            - Competitive salary
            - Health insurance
            - Flexible work hours
            - Professional development opportunities
            """
    
    def generate_job_description_with_resume_skills(self, resume_skills: Set[str], 
                                                  include_percentage: float = 0.7,
                                                  exclude_percentage: float = 0.3) -> str:
        """
        Generate a job description that includes a percentage of skills from the resume
        and excludes others, to test the resume tailoring pipeline.
        
        Args:
            resume_skills: Set of skills from the resume
            include_percentage: Percentage of resume skills to include (0.0 to 1.0)
            exclude_percentage: Percentage of skills to include that are not in the resume (0.0 to 1.0)
            
        Returns:
            A job description with controlled skill inclusion/exclusion
        """
        # Convert resume skills to list for sampling
        resume_skills_list = list(resume_skills)
        
        # Calculate how many skills to include from the resume
        num_include = max(1, int(len(resume_skills_list) * include_percentage))
        skills_to_include = random.sample(resume_skills_list, k=min(num_include, len(resume_skills_list)))
        
        # Calculate how many skills to exclude from the resume
        all_skills = []
        for skill_category in self.SKILLS.values():
            all_skills.extend(skill_category)
        
        # Filter out skills that are in the resume
        non_resume_skills = [skill for skill in all_skills if skill.lower() not in {s.lower() for s in resume_skills_list}]
        
        # Calculate how many non-resume skills to include
        num_exclude = max(1, int(len(resume_skills_list) * exclude_percentage))
        skills_to_exclude = random.sample(non_resume_skills, k=min(num_exclude, len(non_resume_skills)))
        
        # Combine the skills
        all_skills_to_include = skills_to_include + skills_to_exclude
        
        # Generate a job description with these skills
        return self.generate_job_description(skills_to_include=all_skills_to_include)
    
    def extract_skills_from_resume(self, resume_components: Dict[str, str]) -> Set[str]:
        """
        Extract skills from resume components.
        
        Args:
            resume_components: Dictionary of resume components
            
        Returns:
            Set of skills extracted from the resume
        """
        # Combine all resume components
        all_text = " ".join(resume_components.values())
        
        # Create a prompt template
        prompt_template = PromptTemplate(
            template="""
            You are an expert at extracting technical skills and qualifications from resumes.
            
            Extract all the technical skills, tools, technologies, and qualifications from the following resume text.
            Return them as a comma-separated list of individual skills (e.g., "Python, JavaScript, AWS, Docker").
            
            Resume Text:
            {resume_text}
            
            Skills:
            """,
            input_variables=["resume_text"]
        )
        
        # Create the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            # Run the chain
            result = chain.run(resume_text=all_text)
            
            # Parse the result into a set of skills
            skills = {skill.strip() for skill in result.split(",") if skill.strip()}
            return skills
            
        except Exception as e:
            print(f"Error extracting skills from resume: {e}")
            # Return an empty set if the chain fails
            return set()
