import dotenv
from qa_task import qa_task
from inspect_ai import eval
dotenv.load_dotenv()
from prompt import generate_art9_prompt_template, generate_art4_prompt_template, generate_monitor_prompt, generate_art26_prompt_template

CUSTOM_TEMPLATE = open("prompts/monitor_template.txt").read()

biology_art9_prompt = generate_art9_prompt_template("Biology", "Physics")
biology_art4_prompt = generate_art4_prompt_template("Biology", "Physics")
biology_art26_prompt = generate_art26_prompt_template("Biology", "Physics")
biology_monitor = generate_monitor_prompt("Biology")

physics_art9_prompt = generate_art9_prompt_template("Physics", "Chemistry")
physics_art4_prompt = generate_art4_prompt_template("Physics", "Chemistry")
physics_art26_prompt = generate_art26_prompt_template("Physics", "Chemistry")
physics_monitor = generate_monitor_prompt("Physics")

chemistry_art9_prompt = generate_art9_prompt_template("Chemistry", "Biology")
chemistry_art4_prompt = generate_art4_prompt_template("Chemistry", "Biology")
chemistry_art26_prompt = generate_art26_prompt_template("Chemistry", "Biology")
chemistry_monitor = generate_monitor_prompt("Chemistry")

regular_prompt = open("prompts/cotanswer.txt").read()

task_configs = [
    (biology_art9_prompt, biology_monitor, "gpqa_biology"),
    (biology_art4_prompt, biology_monitor, "gpqa_biology"),
    (biology_art26_prompt, biology_monitor, "gpqa_biology"),
    (physics_art9_prompt, physics_monitor, "gpqa_physics"),
    (physics_art4_prompt, physics_monitor, "gpqa_physics"),
    (physics_art26_prompt, physics_monitor, "gpqa_physics"),
    (chemistry_art9_prompt, chemistry_monitor, "gpqa_chemistry"),
    (chemistry_art4_prompt, chemistry_monitor, "gpqa_chemistry"),
    (chemistry_art26_prompt, chemistry_monitor, "gpqa_chemistry"),
    (regular_prompt, biology_monitor, "gpqa_physics"),
    (regular_prompt, physics_monitor, "gpqa_chemistry"),
    (regular_prompt, chemistry_monitor, "gpqa_biology")
]

monitors = [
    # "openai/gpt-4o-mini",
    "openai/gpt-4o",
    # "openai/o3-2025-04-16",
    # "openrouter/openrouter/google/gemini-2.0-flash-001",
    # "openrouter/openrouter/google/gemini-2.5-pro-preview-05-06"
]

# Now create the tasks
tasks = [
    qa_task(
        monitors=monitors,
        main_prompt=prompt,
        monitor_prompt=monitor_instructions, 
        monitor_template=CUSTOM_TEMPLATE,
        main_dataset=dataset,
    ) for prompt, monitor_instructions, dataset in task_configs
]


MODELS=[
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-7-sonnet-20250219",
    "openrouter/openrouter/google/gemini-2.0-flash-001",
]


results = eval(
    tasks,
    model=MODELS,
    max_tasks=24,
    max_connections=100,
    log_level="info",
    log_dir=f"logs/REASONING",
    # reasoning_tokens=8192
)
