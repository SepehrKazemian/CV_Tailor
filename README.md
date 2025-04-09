# Resume Tailoring Pipeline

A smart resume tailoring system that reads job descriptions and matches them with your resume. The pipeline extracts keywords from job descriptions using an LLM and tailors your resume accordingly.

## Features

- Extracts keywords and important information from job descriptions using LLMs (OpenAI, Google Gemini, or Anthropic Claude)
- Tailors your resume based on the job description
- Intelligently substitutes related technologies (e.g., AWS to GCP)
- Ensures the final resume is two pages maximum
- Avoids duplication
- Outputs to Word document (.docx), Markdown, or plain text formats
- Provides warnings when substituting technologies
- Includes test mode to verify resume meets all requirements

## Installation

### Quick Installation (Recommended)

The easiest way to install the package is with the provided installation script:

```bash
# Make the script executable (if not already)
chmod +x install_dev.sh

# Run the installation script
./install_dev.sh
```

This will:
1. Install the package in development mode
2. Check if the required API keys are set
3. Allow you to use the `resume-tailor` command from anywhere

### Manual Installation

Alternatively, you can install the package manually:

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your API keys for the LLM provider you want to use:

- For OpenAI: `export OPENAI_API_KEY=your_api_key_here`
- For Google Gemini: `export GOOGLE_API_KEY=your_api_key_here`
- For Anthropic Claude: `export ANTHROPIC_API_KEY=your_api_key_here`

4. Make the CLI script executable:

```bash
chmod +x resume_tailor_cli.py
```

5. (Optional) Install the package in development mode:

```bash
pip install -e .
```

## Resume Components

The system expects your resume components to be in the `cv/` directory with the following files:

- `ps.txt`: Professional Summary
- `skills.txt`: Skills section
- `experience.txt`: Work Experience section
- `education.txt`: Education section

## Project Structure

The project follows a modular structure:

```
resume_tailor/
├── __init__.py
├── analyzers/
│   └── job_analyzer.py      # Job description analysis
├── tailors/
│   └── resume_tailor.py     # Resume tailoring logic
├── output/
│   └── docx_generator.py    # Word document generation
├── utils/
│   ├── file_utils.py        # File handling utilities
│   └── llm_provider.py      # LLM provider utilities
└── tests/
    └── test_pipeline.py     # Test pipeline
```

## Usage

### Interactive Shell Script (Recommended)

The easiest way to use the resume tailoring pipeline is with the provided shell script:

```bash
# Make the script executable (if not already)
chmod +x run_tailor.sh

# Run the interactive script
./run_tailor.sh
```

The script will:
1. Check for required dependencies and install them if needed
2. Guide you through providing a job description (from a file or direct input)
3. Let you choose which LLM provider to use
4. Prompt for your API key if not set as an environment variable
5. Let you select the output format and filename
6. Run the tailoring process and save the result

### Command-line Usage

You can also use the CLI script directly:

```bash
# Using a job description file
./resume_tailor_cli.py --job-file path/to/job_description.txt

# Using job description text directly
./resume_tailor_cli.py --job-text "Job description text here..."
```

### Advanced Options

```bash
# Specify output file
./resume_tailor_cli.py --job-file job.txt --output my_resume.docx

# Use a different LLM provider
./resume_tailor_cli.py --job-file job.txt --llm google

# Output in a different format
./resume_tailor_cli.py --job-file job.txt --format md

# Provide API key directly
./resume_tailor_cli.py --job-file job.txt --llm anthropic --api-key your_api_key_here

# Run in test mode to verify the resume meets all requirements
./resume_tailor_cli.py --job-file job.txt --test
```

### Command-line Options

- `--job-file`: Path to a file containing the job description
- `--job-text`: Job description text (alternative to --job-file)
- `--output`: Output file path (default: tailored_resume.docx)
- `--format`: Output format: 'docx', 'txt', or 'md' (default: docx)
- `--llm`: LLM provider to use: 'openai', 'google', or 'anthropic' (default: openai)
- `--api-key`: API key for the selected LLM provider
- `--cv-dir`: Directory containing resume component files (default: cv)
- `--test`: Run in test mode to verify the resume meets all requirements

## How It Works

1. The system reads your resume components from the `cv/` directory
2. It analyzes the job description using the selected LLM to extract keywords, required skills, and other important information
3. It tailors each section of your resume based on the analysis:
   - Professional Summary: Emphasizes relevant experience and skills
   - Skills: Prioritizes skills that match the job requirements
   - Work Experience: Highlights relevant projects and achievements
4. It ensures the final resume fits within two pages
5. It saves the tailored resume in the specified format
6. It generates warnings.txt if any technologies were substituted

## Examples

### Using the Example Script (Recommended)

The easiest way to try the pipeline with the example job description:

```bash
# Make the script executable (if not already)
chmod +x run_example.sh

# Run the example script
./run_example.sh
```

This will:
1. Check if all required files exist
2. Prompt you for the LLM provider to use
3. Prompt you for the output format and filename
4. Run the pipeline with the example job description
5. Open the generated resume if it's a Word document

### Using the Interactive Script

For more general usage with any job description:

```bash
./run_tailor.sh
```

Follow the prompts to tailor your resume.

### Using the Command-line

For advanced usage with specific options:

```bash
# Create a tailored resume for a Data Science job
./resume_tailor_cli.py --job-file example_job.txt --output data_science_resume.docx --llm openai

# Use Google's Gemini model
./resume_tailor_cli.py --job-file example_job.txt --llm google --output gemini_resume.docx

# Output as markdown
./resume_tailor_cli.py --job-file example_job.txt --format md --output resume.md

# Run in test mode
./resume_tailor_cli.py --job-file example_job.txt --test
```

An example job description file (`example_job.txt`) is included in the repository to help you get started.

## Testing

### Comprehensive Testing (Recommended)

The easiest way to run all tests is with the provided test script:

```bash
# Make the script executable (if not already)
chmod +x run_tests.sh

# Run all tests
./run_tests.sh
```

This will:
1. Check if all required files exist
2. Run the test pipeline with the example job description
3. Run pytest on all test modules
4. Test with each LLM provider that has an API key set
5. Verify that all tests pass

### Quick Test

For a quick test of the pipeline:

```bash
# Make the script executable (if not already)
chmod +x test_resume.py

# Run the test script
./test_resume.py
```

This will:
1. Check if all required files exist
2. Run the test pipeline with the example job description
3. Generate a tailored resume
4. Verify that it meets all requirements
5. Save the test results to a JSON file

### Manual Test

You can also run the test pipeline manually:

```bash
./resume_tailor_cli.py --job-file example_job.txt --test
```

This will:
1. Analyze the job description
2. Generate a tailored resume
3. Verify that it has all required sections
4. Check that it's not more than 2 pages
5. Ensure it contains all important keywords from the job description
6. Generate warnings for any technology substitutions

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
