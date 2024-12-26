import sys
import ast
from pathlib import Path
import pandas as pd

CUR_DIR = Path(__file__).parent.resolve()
added_path = CUR_DIR.parent.resolve()
print(added_path)
sys.path.insert(0, str(added_path))

from src import common_path


TASKS_SHOW_MAP = {
    "Model Quantization, Pruning, or Distillation": "Model Compression",
    "Reinforcement Learning from Human Feedback (RLHF)": "RLHF",
    "Prompt Engineering Techniques": "Prompt Engineering",
    "API recommendation for coding": "API recommendation",
    "Testing Strategies for FM/LLM/GenAI Models": "Testing Strategies",
    "Platforms/Tools/Studios for FM/LLM/GenAI": "Platforms/Tools/Studios",
    "Specialized Databases for FM/LLM/GenAI": "Specialized Databases",
    "Workflow Orchestration for FM/LLM/GenAI": "Workflow Orchestration",
    "FM/LLM/GenAI Agent, Copilot, or Assistant": "AI Agent",
    "Full Parameter Fine-Tuning of a Pre-Trained Model": "General Fine-Tuning",
    "RAG for FM/LLM/GenAI": "RAG",
    "Model Deployment on Device": "Model Deployment on Local",
    "Low-Rank Adaptation (LoRA) for Foundation Models": "LoRA",
    "Guardrails for FM/LLM/GenAI": "Guardrails",
}

ACTIVITY_SHOW_MAP = {
    "Model Fine-Tuning": "Model Customization",
}

def format_title_link(row):
    """Combine title and link into Markdown URL format."""
    title = row['title'].replace('|', '\|')
    return f"[{title}]({row['link']})"


# def visualize_tasks(tasks):
#     """Convert list of tasks into HTML-styled badges."""
#     tasks = [TASKS_SHOW_MAP[task] if task in TASKS_SHOW_MAP else task for task in tasks]
#     return ', '.join(
#         [f'<span style="background-color:#d1ecf1; color:#0c5460; padding:2px 4px; border-radius:4px;">{task}</span>'
#          for task in tasks])


def visualize_tasks(tasks):
    """Convert list of tasks into a comma-separated string."""
    tasks = [TASKS_SHOW_MAP[task] if task in TASKS_SHOW_MAP else task for task in tasks]
    return ' '.join(["`[" + task + "]`" for task in tasks])

def generate_toc(activities):
    """Generate a Markdown Table of Contents with links to each activity section."""
    toc = "# Table of Contents\n\n"
    toc += '\n'.join([f"- [{activity}](#{activity.lower().replace(' ', '-')})" for activity in activities])
    toc += "\n\n"
    return toc

def convert_to_markdown(area):
    assert area in ["FM4SE", "SE4FM"], "Invalid area"
    # Load the data
    df = pd.read_csv(common_path.DATA_PATH / f"{area}_activities.csv")

    # Select only the required columns
    df = df[['id', 'activity', 'tasks', 'title', 'link']]
    df['activity'] = df['activity'].apply(lambda x: x[x.find(". ") + 2:])

    # Merge title and link into a Markdown URL format
    df['post'] = df.apply(format_title_link, axis=1)

    # Convert tasks into comma-separated tags
    df['tasks'] = df['tasks'].apply(ast.literal_eval)
    df['tasks'] = df['tasks'].apply(lambda x: [task[task.find(". ") + 2:] for task in x])
    df['tags'] = df['tasks'].apply(visualize_tasks)

    # Drop the original title and link columns (no longer needed)
    df = df[['id', 'activity', 'tags', 'post']]
    df['activity'] = df['activity'].apply(lambda x: ACTIVITY_SHOW_MAP[x] if x in ACTIVITY_SHOW_MAP else x)

    # Group by activity and sort each group by id
    grouped = df.sort_values(by='id').groupby('activity')

    # Print each group as a Markdown table
    for activity, group in grouped:
        print(f"## {activity}\n")
        print(group[['id', 'tags', 'post']].to_markdown(index=False))
        print("\n")

    # Create a list of unique activities for the Table of Contents (ToC)
    activities = df['activity'].unique()

    # Open a Markdown file to write the output
    with open(f"../posts_index/{area}.md", "w") as f:
        # Write the Table of Contents at the top of the file
        f.write(generate_toc(activities))

        # Iterate over each group and write to the file
        for activity, group in grouped:
            f.write(f"## {activity}\n\n")  # Write the activity name as a heading
            f.write(group[["id", 'post', 'tags']].to_markdown(index=False))  # Write the table as Markdown
            f.write("\n\n")  # Add some spacing between tables


def main():
    convert_to_markdown("FM4SE")
    convert_to_markdown("SE4FM")


if __name__ == '__main__':
    main()
