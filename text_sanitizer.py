import re

def load_sanitized_list(filepath):
    """
    Loads a list of replacement pairs from a file.

    Each line in the file should contain a pair of strings separated by a colon.
    These pairs are used for text sanitization.

    Args:
        filepath (str): The path to the file containing the replacement pairs.

    Returns:
        list of tuple: A list of tuples where each tuple contains two strings.
    """
    replacements = []
    with open(filepath, 'r') as file:
        for line in file:
            if ':' in line:
                a, b = line.strip().split(':')
                replacements.append((a, b))
    return replacements

def sanitizer(text, replacements):
    """
    Replaces all occurrences of each 'A' with 'B' in the given text.

    Args:
        text (str): The text to be sanitized.
        replacements (list of tuple): A list of tuples where each tuple contains two strings (A, B).

    Returns:
        str: The sanitized text.
    """
    for a, b in replacements:
        text = re.sub(re.escape(a), b, text, flags=re.IGNORECASE)
    return text

def unsanitizer(text, replacements):
    """
    Replaces all occurrences of each 'B' with 'A' in the given text.

    Args:
        text (str): The text to be unsanitized.
        replacements (list of tuple): A list of tuples where each tuple contains two strings (A, B).

    Returns:
        str: The unsanitized text.
    """
    for a, b in replacements:
        text = re.sub(re.escape(b), a, text, flags=re.IGNORECASE)
    return text

# Example usage:
# replacements = load_sanitized_list('rhizome/sanitized_list.md')
# sanitized_text = sanitizer("some text with A", replacements)
# unsanitized_text = unsanitizer(sanitized_text, replacements)

pass