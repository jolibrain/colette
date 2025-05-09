from copy import deepcopy
from pprint import pprint


def pretty_print_response(response: dict):
    truncated_response = deepcopy(response)
    nodes = [truncated_response]

    while nodes:
        current = nodes.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if isinstance(value, str) and value.startswith("data:image/"):
                    current[key] = value[:45]
                elif isinstance(value, (dict, list)):
                    nodes.append(value)
        elif isinstance(current, list):
            for i in range(len(current)):
                item = current[i]
                if isinstance(item, str) and item.startswith("data:image/"):
                    current[i] = item[:45]
                elif isinstance(item, (dict, list)):
                    nodes.append(item)

    pprint(truncated_response)


def compare_dicts(dict1: dict, dict2:dict, path:str=""):
    differences = []

    for key in dict1:
        current_path = f"{path}/{key}" if path else key

        if key not in dict2:
            differences.append(f"Key '{current_path}' found in 'dict1' but missing in 'dict2'.")
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(compare_dicts(dict1[key], dict2[key], current_path))
            elif dict1[key] != dict2[key]:
                differences.append(
                    f"Value mismatch at '{current_path}': 'dict1' has {dict1[key]}, 'dict2' has {dict2[key]}"
                )

    return differences
