import os
import ast
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
from google import genai


MODEL_NAME = "gemini-3.5-flash-lite"

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

WORK_DIR = Path(tempfile.gettempdir()) / "multi_agent_test_generator"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def ask_gemini(instructions, content):
    if client is None:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it in Render "
            "Dashboard > Environment."
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"{instructions}\n\nInput:\n{content}",
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


def code_analyzer(source_code):
    instructions = """
You are the Code Analyzer agent.

Analyze the supplied Python source code.
Explain:
- Purpose of the code
- Functions and their responsibilities
- Inputs and outputs
- Edge cases
- Possible errors
- Important behavior

Do not execute the supplied code.
"""
    return ask_gemini(instructions, source_code)



def test_planner(source_code, analysis):
    instructions = """
You are the Test Planner agent.

Create a detailed pytest test plan for the source code.
Include normal cases, boundary cases, invalid inputs,
edge cases, and expected behavior.

Do not invent functions that do not exist in the source.
"""
    content = (
        f"SOURCE CODE:\n{source_code}\n\n"
        f"CODE ANALYSIS:\n{analysis}"
    )
    return ask_gemini(instructions, content)


def test_generator(source_code, plan):
    instructions = """
You are the Test Generator agent.

Generate executable pytest tests for the source code.

Rules:
- Return Python code only.
- Do not use Markdown code fences.
- Assume the source is saved as source_code.py.
- Import functions from source_code when appropriate.
- Do not invent functions or classes.
- Include useful assertions and edge cases.
- Do not execute the source code.
- Avoid network access and destructive operations.
"""
    content = (
        f"SOURCE CODE:\n{source_code}\n\n"
        f"TEST PLAN:\n{plan}"
    )
    return ask_gemini(instructions, content)

def test_reviewer(source_code, plan, generated_tests):
    instructions = """
You are the Test Reviewer agent.

Review the generated pytest tests against the source code
and test plan.

Check:
- Incorrect imports
- Nonexistent functions
- Incorrect expected values
- Missing edge cases
- Syntax errors
- Weak assertions
- Unsafe operations

Return an overall review, problems found,
recommended corrections, and a revised pytest code section.

Do not claim tests passed unless they were actually run.
"""
    content = (
        f"SOURCE CODE:\n{source_code}\n\n"
        f"TEST PLAN:\n{plan}\n\n"
        f"GENERATED TESTS:\n{generated_tests}"
    )
    return ask_gemini(instructions, content)



def run_workflow(source_code):
    if not source_code or not source_code.strip():
        raise gr.Error("Please enter Python source code.")


    try:
        ast.parse(source_code)
    except SyntaxError as exc:
        return (
            f"REJECTED: Invalid Python syntax. "
            f"Line {exc.lineno}: {exc.msg}",
            "",
            "",
            "",
            "",
            None,
            None,
        )

    try:
        analysis = code_analyzer(source_code)
        plan = test_planner(source_code, analysis)
        generated_tests = test_generator(source_code, plan)
        review = test_reviewer(source_code, plan, generated_tests)

        WORK_DIR.mkdir(parents=True, exist_ok=True)

        source_path = WORK_DIR / "source_code.py"
        tests_path = WORK_DIR / "test_generated.py"
        analysis_path = WORK_DIR / "analysis.txt"
        plan_path = WORK_DIR / "test_plan.txt"
        review_path = WORK_DIR / "test_review.txt"
        zip_path = WORK_DIR / "multi_agent_test_project.zip"

        source_path.write_text(source_code, encoding="utf-8")
        tests_path.write_text(generated_tests, encoding="utf-8")
        analysis_path.write_text(analysis, encoding="utf-8")
        plan_path.write_text(plan, encoding="utf-8")
        review_path.write_text(review, encoding="utf-8")

        with zipfile.ZipFile(
            zip_path, "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for file_path in [
                source_path,
                tests_path,
                analysis_path,
                plan_path,
                review_path,
            ]:
                archive.write(file_path, arcname=file_path.name)

        return (
            "COMPLETED: All four agents finished successfully.",
            analysis,
            plan,
            generated_tests,
            review,
            str(tests_path),
            str(zip_path),
        )

    except Exception as exc:
        raise gr.Error(
            f"Workflow failed: {exc}. Check your Gemini API key, "
            "model availability, and Render logs."
        )



with gr.Blocks(
    title="Multi-Agent Test Generation Agent"
) as app:

    gr.Markdown(
        """
        # Multi-Agent Test Generation Agent

        Submit Python code to generate and review automated tests.

        **Agents**
        1. Code Analyzer
        2. Test Planner
        3. Test Generator
        4. Test Reviewer

        Python syntax is checked before processing.
        The supplied code is not executed by this application.
        """
    )

    source_input = gr.Code(
        label="Python Source Code",
        language="python",
        lines=15,
        value="def add(a, b):\n    return a + b\n",
    )

    run_button = gr.Button(
        "Generate Tests",
        variant="primary",
    )

    status_output = gr.Textbox(label="Workflow Status")
    analysis_output = gr.Textbox(
        label="Agent 1 - Code Analysis",
        lines=8,
    )
    plan_output = gr.Textbox(
        label="Agent 2 - Test Plan",
        lines=8,
    )
    tests_output = gr.Code(
        label="Agent 3 - Generated Tests",
        language="python",
    )
    review_output = gr.Textbox(
        label="Agent 4 - Test Review",
        lines=10,
    )

    with gr.Row():
        test_download = gr.File(
            label="Download Generated Tests"
        )
        project_download = gr.File(
            label="Download Complete Project ZIP"
        )

    run_button.click(
        fn=run_workflow,
        inputs=[source_input],
        outputs=[
            status_output,
            analysis_output,
            plan_output,
            tests_output,
            review_output,
            test_download,
            project_download,
        ],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
