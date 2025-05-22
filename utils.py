import os
from ollama import chat, ChatResponse
import json
import re

def remove_brackets_from_files(directory):
    """
    For each .md file in the given folder, remove double enclosed quotes (foam backlinks)
    """
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Only process files (not directories)
        if os.path.isfile(filepath) and filename.endswith('.md'):
            # Read file contents
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            # Remove  and  characters
            new_content = content.replace('[', '').replace(']', '')
            # Write the modified content back to the file
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Processed: {filename}")

def remove_yaml_frontmatter(directory):
    """
    For each .md file in the given folder:
    - Remove all lines before the first line that starts with a pound sign (#).
    - If no such line exists, and any line contains '---', remove all lines up to and including the last such line.
    - Otherwise, leave the file unchanged.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Find the index of the first line starting with '#'
                header_index = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#")), None)
                if header_index is not None:
                    # Remove all lines before the first header
                    new_lines = lines[header_index:]
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                else:
                    # Find the last index of a line containing '---'
                    dash_indices = [i for i, line in enumerate(lines) if '---' in line]
                    if dash_indices:
                        last_dash_index = dash_indices[-1]
                        new_lines = lines[last_dash_index + 1:]
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)
                        print(f"Removed lines up to last '---' in {filename}.")
                    else:
                        print(f"No header or '---' found in {filename}; file left unchanged.")


def extract_names_from_md_files(directory):
    """Extracts people names from markdown files using llms"""
    results = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.md'):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # LLM processing
            prompt = f"Extract names of people mentioned in this note, and in your response list each of the names in double brackets as [[A]], [[B]], etc. Only use exact names of people, not descriptions, and only return the list without any other words in your response as a comma separated list. don't include the name (I) ,if there are no names return a blank string, and don't include abstract topics:\n\n{content}"
            response: ChatResponse = chat(
                model='gemma3:12b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            results[filename] = response.message.content
            
    return results

def extract_themes_from_md_files(directory):
    """Extracts themes from markdown files using llms"""
    results = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.md'):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # LLM processing
            prompt = f"Extract themes mentioned in this note, and in your response list each of the themes in double brackets as [[A]], [[B]], etc. An Example themes include, but are not limited to: loneliness, unity, addiction, sex, love, science, philosophy. Keep all themes lowercase and as single words. :\n\n{content}"
            response: ChatResponse = chat(
                model='gemma3:12b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            results[filename] = response.message.content
            
    return results

def modify_yaml_frontmatter(json_path, base_dir, key):
    """
    Reads a JSON file containing {'filename': 'ABC'} entries,
    checks if the YAML frontmatter exists in each file,
    and inserts or modifies the key and its associated value.

    The frontmatter is expected to end with '---' (no opening '---').

    Args:
        json_path (str): Path to the JSON file with filenames.
        base_dir (str): Directory where the files are located.
        key (str): Name of the entry to add/modify in the frontmatter.
    """
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for filename, people_value in data.items():
        file_path = os.path.join(base_dir, filename)

        # Read the existing content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regular expression to detect the YAML frontmatter (ending with ---)
        frontmatter_pattern = re.compile(r"^(.*?)(?:\n)?---\s*", re.DOTALL)
        match = frontmatter_pattern.match(content)

        # If frontmatter exists, modify it
        if match:
            frontmatter = match.group(1)  # Extract content before ending ---
            frontmatter_lines = frontmatter.splitlines()

            # Remove any existing key
            frontmatter_lines = [
                line for line in frontmatter_lines if not line.strip().startswith(f"{key}:")
            ]

            # Add the new key-value pair (at the top)
            frontmatter_lines.insert(0, f"{key}: {people_value}")

            # Rebuild the YAML frontmatter (no opening ---)
            new_frontmatter = "\n".join(frontmatter_lines) + "\n---\n"

            # Replace the old frontmatter with the new one in the content
            new_content = frontmatter_pattern.sub(new_frontmatter, content, count=1)

        else:
            # If no frontmatter exists, create one (no opening ---)
            new_frontmatter = f"{key}: {people_value}\n---\n"
            new_content = new_frontmatter + content

        # Write back the updated content to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)

# Usage example:
directory_path = '/home/ego/daily-backup'
remove_brackets_from_files(directory_path)
remove_yaml_frontmatter(directory_path)
names = extract_names_from_md_files(directory_path)
with open("names.json","w") as file:
    json.dump(names,file)
themes = extract_themes_from_md_files(directory_path)
with open("themes.json","w") as file:
    json.dump(themes,file)
modify_yaml_frontmatter(
    json_path = "/home/ego/projects/journal-utils/names-revised.json",
    base_dir = directory_path,
    key = "people"
)
modify_yaml_frontmatter(
    json_path = "/home/ego/projects/journal-utils/themes-revised.json",
    base_dir = directory_path,
    key = "themes"
)
