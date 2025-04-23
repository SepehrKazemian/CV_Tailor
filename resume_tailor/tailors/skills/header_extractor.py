def extract_all_headers(raw_client, model_name, job_description):
    tool_name = "extract_headers"

    prompt = (
        "# Resume Skill Section Header Generator\n\n"
        "You are a resume intelligence assistant. Your task is to generate professional and well-structured "
        "resume skill section headers based on the content and expectations described in a job description.\n\n"
        "## Objective:\n"
        "Create a list of clean, professional **skill section headers** that would be appropriate for organizing the candidate’s skills on a resume.\n\n"
        "## Guidelines:\n"
        "- Headers must group **related skills or tools** under a common category\n"
        "- Avoid overly broad terms like 'General Skills', 'Miscellaneous', or 'Abilities'\n"
        "- Avoid merged or compound phrases like 'A & B' or slashes like 'A/B'.\n"
        "- Do not include soft skills, personal traits, or generic attributes\n"
        "- A header **can be a single-topic umbrella** even if it maps to one small area\n"
        "## Examples from Various Domains (not exhaustive):\n"
        "- Education: 'Instructional Design', 'Classroom Technology'\n"
        "- Finance: 'Risk Modeling', 'Compliance Tools'\n"
        "- Healthcare: 'Patient Information Systems', 'Clinical Operations'\n"
        "- Retail: 'Inventory Systems', 'Point-of-Sale Platforms'\n"
        "- Marketing: 'Digital Campaign Tools', 'Analytics Platforms'\n"
        "- Logistics: 'Fleet Management Systems', 'Warehouse Technology'\n\n"
        "NOTE: The extracted headers must be single area coverage and as complete as possible."
        "## Output Format:\n"
        "Return only a JSON object with a single key `headers` that maps to an array of strings.\n\n"
        "Job Description:\n"
        f"{job_description}\n\n"
        "Your Response:\n"
    )

    desc = "Extract professional skill section headers from a job description"
    schema = {
        "type": "object",
        "properties": {
            "headers": {
                "type": "array",
                "description": "Concise, professional section headers that group related skills. Each header should be specific, meaningful, and reflect a distinct competency area.",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["headers"]
    }
    tool_schema = [{
        "name": tool_name,
        "description": desc,
        "input_schema": schema
    }]

    # Run the tool-call-enabled LLM
    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        max_tokens=1024,
        temperature=0.3,
        tool_choice={"type": "tool", "name": tool_name}
    )

    return response.content[0].input["headers"]