import ast
import json
from pathlib import Path
from typing import Any, Dict

try:
    from .text_event_parser import read_marshaled_line
except ImportError:
    from text_event_parser import read_marshaled_line


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a legacy nested ETW event record into a single-level dictionary.

    Expected pattern:
    - the first top-level payload may itself be a nested dict
    - EventDescriptor is expanded into top-level keys
    """
    flattened: Dict[str, Any] = {}
    first_item = True

    for key, value in record.items():
        if first_item and isinstance(value, dict):
            first_item = False
            for nested_key, nested_value in value.items():
                if nested_key == "EventDescriptor" and isinstance(nested_value, dict):
                    for event_key, event_value in nested_value.items():
                        flattened[event_key] = event_value
                else:
                    flattened[nested_key] = nested_value
        else:
            first_item = False
            flattened[key] = value

    return flattened


def flatten_json_lines_file(input_path: str | Path, output_path: str | Path) -> None:
    """
    Read legacy marshaled text entries and write flattened JSON-lines output.

    The original format stores each event as a Python-literal dictionary string,
    so ast.literal_eval is used here rather than eval.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8", errors="ignore") as src, output_path.open(
        "w", encoding="utf-8", errors="ignore"
    ) as dst:
        data = read_marshaled_line(src)

        while data:
            record = ast.literal_eval(data)
            data = read_marshaled_line(src)

            flattened = flatten_record(record)
            serialized = json.dumps(flattened)

            if data:
                dst.write(serialized + "\n")
            else:
                dst.write(serialized)


def main() -> None:
    raise SystemExit("Use flatten_json_lines_file(input_path, output_path).")


if __name__ == "__main__":
    main()
