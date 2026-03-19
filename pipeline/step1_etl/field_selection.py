import json
from pathlib import Path
from typing import Any, Dict

try:
    from .text_event_parser import EXCLUDED_ATTRIBUTES, read_marshaled_line
except ImportError:
    from text_event_parser import EXCLUDED_ATTRIBUTES, read_marshaled_line


THREAD_ID_FIELDS = {"ThreadId", "ThreadID"}


def _normalize_thread_id(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return int(value, 16)
        except ValueError:
            return value
    return value


def filter_selected_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove unwanted attributes and normalize selected fields.
    """
    filtered: Dict[str, Any] = {}

    for key, value in record.items():
        if key in EXCLUDED_ATTRIBUTES:
            continue

        if key in THREAD_ID_FIELDS:
            filtered[key] = _normalize_thread_id(value)
        else:
            filtered[key] = value

    return filtered


def select_fields_from_json_lines(input_path: str | Path, output_path: str | Path) -> None:
    """
    Read one legacy ETW record per line and write a JSON array of filtered records.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8", errors="ignore") as src, output_path.open(
        "w", encoding="utf-8", errors="ignore"
    ) as dst:
        data = read_marshaled_line(src)
        dst.write("[")

        while data:
            record = json.loads(data)
            data = read_marshaled_line(src)

            filtered = filter_selected_fields(record)
            serialized = json.dumps(filtered)

            if data:
                dst.write(serialized + ",\n")
            else:
                dst.write(serialized + "]")


def main() -> None:
    raise SystemExit("Use select_fields_from_json_lines(input_path, output_path).")


if __name__ == "__main__":
    main()
