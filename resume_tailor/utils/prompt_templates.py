PROFESSIONAL_SUMMARY_PROMPT = """
You are a skilled resume writer and career branding expert.

Your task is to rewrite the candidate’s professional summary into a sharper, more targeted version
that aligns directly with the job posting. The output must be **no more than 4 lines**, confident
in tone, and free of vague or generic phrases.

Use the following job-specific insights and resume data to guide your summary:

1. Industry: {business}
2. Key responsibilities for this role: {responsibilities}
3. Critical keywords and technologies to highlight: {keywords}, {tech_stack}
4. Experience focus and domain knowledge: {experience_focus}, {domain_knowledge}
5. Relevant soft skills: {soft_skills}

Use this information to write a 4-line summary that:
- Shows how the candidate is uniquely valuable to this role
- Includes technical strengths *and* business impact
- Clearly conveys experience, expertise, and real outcomes
- Feels tailored — not generic — to this job
- Matches the suggested tone: Confident, precise, outcome-oriented. Prioritize technologies,
  frameworks, and architecture impact. Avoid vague business lingo. Use more innovation-oriented
  tone and less comma-heavy tight rhythm.

Here is the original professional summary:
{original_summary}

Now write the final version. Only return the revised 4-line summary — no commentary or extra explanation.
"""

REWRITE_SUMMARY_PROMPT = """
You are a skilled resume writer. Rewrite the following professional summary based on the
provided feedback, ensuring it remains under 5 lines and adheres to best practices
(confident, precise, outcome-oriented, tailored).

Original Summary:
{original_summary}

Feedback:
{feedback}

Return only the rewritten professional summary.
"""

# --- Skills Refinement Prompts ---

STEP1_EXTRACT_SKILLS_PROMPT = """
Given the job description and the candidate's skills list below, identify relevant skills for the specific header: "{header}"

Job Description:
{job_description}

Candidate's Skills List (skills.txt):
{original_skills_text}

Instructions:
1. List skills explicitly mentioned or strongly implied in the job description that fit under the header "{header}". These are "job_skills".
2. List skills from the candidate's list that match the job requirements AND fit under the header "{header}". These are "candidate_skills".
3. If no skills fit under the header for either category, return an empty list for that category.
4. Output ONLY a valid JSON object with keys "job_skills" and "candidate_skills", each containing a list of strings.
   Example: {{"job_skills": ["Skill A", "Skill B"], "candidate_skills": ["Skill C"]}}
"""

STEP2_SELECT_HEADERS_PROMPT = """
You are an expert resume analyst. Given the following JSON data containing skills extracted under various predefined headers, your task is to select the **4 most relevant technical headers** for this specific job and **ALWAYS include the "Soft Skills" header**.

Prioritize technical headers based on the number and relevance of skills listed under both 'job_skills' (from the job description) and 'candidate_skills' (from the CV). Headers with more relevant skills are generally better choices.

Full Available Skills Data per Header (JSON):
{all_skills_data_json}

Instructions:
1. Analyze the relevance of each technical header based on the skills listed within it (both job_skills and candidate_skills).
2. Select the **top 4 most relevant technical headers**.
3. **Crucially, you MUST include the "Soft Skills" header and its corresponding data** from the input JSON, regardless of its content relevance score compared to technical skills.
4. Construct a new JSON object containing exactly 5 keys: the 4 selected technical headers and the mandatory "Soft Skills" header.
5. The value for each key in the output JSON must be the **complete skill data object** (including both "job_skills" and "candidate_skills" lists) as provided in the input 'Full Available Skills Data per Header'. Do not omit or alter the skill lists for the selected headers.
6. For each of the 5 selected headers, try to keep it less than 100 characters.
7. Ensure the final output is a **single, valid JSON object** with exactly 5 key-value pairs.


Example Output Format:
{{
  "Relevant Technical Header 1": [...],
  "Relevant Technical Header 2": [...],
  "Relevant Technical Header 3": [...],
  "Relevant Technical Header 4": [...],
  "Soft Skills": [...] // Must be included with its data
}}

Output ONLY the valid JSON object.
"""

STEP4_REFINE_HEADERS_PROMPT = """
You are an expert resume analyst. Based on the judge's feedback, refine the selected skills section headers and their content. The goal is to address the feedback while maintaining exactly 5 headers (4 technical + "Soft Skills").

Previous Skills Section Structure (JSON):
{previous_structure_json}

Judge's Feedback:
{feedback}

Full Available Skills Data per Header (JSON) - Use this to find alternative skills or headers if needed:
{all_skills_data_json}

Instructions:
1. Carefully consider the feedback and make necessary corrections to the 'Previous Skills Section Structure'. This might involve:
   - Adding important 'job_skills' (from the 'Full Available Skills Data') that were missing under the correct header.
   - Removing irrelevant 'candidate_skills'.
   - Moving skills to a more appropriate header if the feedback suggests miscategorization.
   - Replacing a less relevant technical header (from the 'Previous Skills Section Structure') with a more relevant one (choose from the 'Full Available Skills Data').
   - Rewording/summarizing skills if feedback mentions wordiness (though focus is structure/content).
2. **Ensure the "Soft Skills" header is ALWAYS present in the output and contains its corresponding data from the 'Full Available Skills Data'.** Do NOT remove or replace the "Soft Skills" header itself.
3. The final output must contain exactly 5 headers (the refined 4 technical + "Soft Skills").
4. Output ONLY the refined, valid JSON object with the 5 headers and their updated skill data (job_skills and candidate_skills lists). Use the same format as the 'Previous Skills Section Structure'.
"""

# --- Experience Refinement Prompts ---

EXP_STEP1_TAILOR_BULLETS_PROMPT = """
You are an expert resume writer tailoring experience bullet points for a specific job.

Job Description:
{job_description}

Job Analysis Insights:
- Keywords: {keywords}
- Tech Stack: {tech_stack}
- Experience Focus: {experience_focus}
- Tone: {tone}

Original Bullet Points for this specific project/job:
{original_bullets_str}

Instructions:
1. Select only the bullet points most relevant to the job description and analysis insights provided above. Discard irrelevant points entirely.
2. Rewrite the selected bullet points to:
   - Align closely with the job requirements and keywords. Use terminology from the job description where appropriate.
   - Be concise and impactful (strictly maximum 2 lines each). Start with strong action verbs.
   - Quantify achievements with metrics or specific outcomes whenever possible.
   - Use a {tone} tone (e.g., technical, innovative, business-focused).
   - Substitute technologies mentioned in the original bullet points with equivalent technologies from the job's tech stack ({tech_stack}) ONLY if appropriate and the accomplishment is transferable (e.g., replace GCP with AWS if the job requires AWS and the task is cloud-agnostic). Be cautious with substitutions; only do it if it makes sense.
3. Output ONLY a Python-style list of strings, where each string is a rewritten bullet point.
   Example: ["Rewrote bullet 1 focusing on AWS.", "Quantified impact for bullet 2.", "Selected and reworded bullet 4."]
   Return an empty list `[]` if no original bullet points are relevant.
"""

# Note: EXP_STEP2_JUDGE_PROMPT is implicitly defined by the user content passed to the tool call
# in ExperienceRefiner._step2_validate_bullets

# --- Other Prompts ---

SKILLS_PROMPT = """
You are a skilled resume writer.

Your task is to tailor the candidate's skills section to align with a specific job posting by
emphasizing the most relevant technical skills, tools, techniques, and soft skills.

This job emphasizes:
- Keywords: {keywords}
- Tech stack: {tech_stack}
- Related technologies: {related_technologies}
- Company Industry: {business}
- Soft skills: {soft_skills}
- Missing but important skills to include: {missing_keywords}

Original skills:
{original_skills}

Instructions:
1. Organize skills into clear, scannable categories (e.g., LLM Ecosystem, Cloud & DevOps,
   ML Frameworks, Data Engineering, etc.)
2. Keep the bullet point format (starting with •) for each category
3. Prioritize skills that match the job description, including missing but relevant ones the
   candidate hasn't listed yet
4. Include not just tools, but also techniques (e.g., RAG, forecasting), practices (e.g., CI/CD,
   observability), and deployment patterns (e.g., FastAPI, Docker)
5. Avoid generic or vague language — be precise and concise
6. Avoid listing the same tool/technology in multiple categories
7. Return only up to 4-5 bullet points unless absolutely necessary
8. Do not use full sentences — this is a skills section
9. Preserve the formatting style and clarity

Return only the revised skills section, starting directly with bullet points. Do not include any
commentary or headers.
""" # Note: This prompt seems related to the OLD skills tailoring logic. Keep for now but likely unused by SkillsRefiner.

EXPERIENCE_PROMPT = """
You are a skilled resume writer.

Tailor this work experience section for a job that emphasizes:
- Keywords: {keywords}
- Tech stack: {tech_stack}
- Experience focus: {experience_focus}
- Domain knowledge: {domain_knowledge}

Technology substitutions to highlight:
{substitutions}

Original experience:
{original_experience}

Provide a tailored experience section that:
1. Emphasizes projects and responsibilities most relevant to the target job
2. Highlights achievements that demonstrate skills mentioned in the job description
3. Adjusts technical terminology to match the job description where appropriate
4. Maintains the same formatting and structure as the original
5. Prioritizes the most relevant experiences
6. Keeps the section concise while preserving important details
7. Intelligently substitutes technologies when appropriate (e.g., if the job mentions AWS but the
   resume has GCP experience)
8. Highlights the substituted technologies mentioned above

Return only the revised experience text.
""" # Note: This prompt seems related to the OLD experience tailoring logic. Keep for now but likely unused by ExperienceRefiner.

PAGE_LIMIT_PROMPT = """
You are a skilled resume editor.

The following resume is too long (exceeds 2 pages). Please condense it while preserving the most
important information:

{resume}

Guidelines for condensing:
1. Maintain all section headers (Professional Summary, Skills, Work Experience, Education)
2. Preserve the most relevant skills and experiences for the job
3. Remove redundant or less important details
4. Shorten descriptions while maintaining key achievements
5. Keep the same overall structure and formatting
6. Ensure the final resume fits within 2 pages (approximately {max_chars} characters)

Return only the condensed resume.
"""
