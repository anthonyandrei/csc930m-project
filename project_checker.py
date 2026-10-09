import argparse
import json
import re
from pathlib import Path


MAX_TOTAL_MARKDOWN_WORDS = 6000
MAX_WORDS_PER_MARKDOWN = 100
MAX_CODE_LINES = 30
MAX_FIGURES = 15


def count_words(text):
    return len(re.findall(r"\S+", text))


def count_code_lines(source):
    return sum(1 for line in source.splitlines() if line.strip())


def has_comment(code):
    for line in code.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            return True

        code_without_string = re.sub(
            r"(\"\"\".*?\"\"\"|'''.*?''')",
            "",
            stripped,
            flags=re.DOTALL,
        )

        in_single = False
        in_double = False
        escaped = False

        for i, char in enumerate(code_without_string):
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == "'" and not in_double:
                in_single = not in_single
                continue

            if char == '"' and not in_single:
                in_double = not in_double
                continue

            if char == "#" and not in_single and not in_double:
                return True

    return False


def count_figures(notebook):
    figure_count = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        for output in cell.get("outputs", []):
            output_type = output.get("output_type", "")

            if output_type == "display_data" or output_type == "execute_result":
                data = output.get("data", {})

                image_types = {
                    "image/png",
                    "image/jpeg",
                    "image/jpg",
                    "image/svg+xml",
                    "image/webp",
                }

                figure_count += sum(
                    1 for image_type in image_types if image_type in data
                )

            elif output_type == "display_data":
                figure_count += 1

    return figure_count


def check_notebook(ipynb_path):
    with open(ipynb_path, "r", encoding="utf-8") as file:
        notebook = json.load(file)

    cells = notebook.get("cells", [])

    total_markdown_words = 0
    markdown_violations = []
    code_line_violations = []
    comment_violations = []

    markdown_cell_number = 0
    code_cell_number = 0

    for index, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            markdown_cell_number += 1

            word_count = count_words(source)
            total_markdown_words += word_count

            if word_count > MAX_WORDS_PER_MARKDOWN:
                markdown_violations.append(
                    {
                        "cell": index,
                        "markdown_number": markdown_cell_number,
                        "words": word_count,
                    }
                )

        elif cell_type == "code":
            code_cell_number += 1

            line_count = count_code_lines(source)

            if line_count > MAX_CODE_LINES:
                code_line_violations.append(
                    {
                        "cell": index,
                        "code_number": code_cell_number,
                        "lines": line_count,
                    }
                )

            if has_comment(source):
                comment_violations.append(
                    {
                        "cell": index,
                        "code_number": code_cell_number,
                    }
                )

    figure_count = count_figures(notebook)

    total_words_violation = total_markdown_words > MAX_TOTAL_MARKDOWN_WORDS
    figure_violation = figure_count > MAX_FIGURES

    has_violations = (
        total_words_violation
        or len(markdown_violations) > 0
        or len(code_line_violations) > 0
        or figure_violation
        or len(comment_violations) > 0
    )

    print("\n" + "=" * 60)
    print("NOTEBOOK RESTRICTION CHECK")
    print("=" * 60)
    print(f"File: {Path(ipynb_path).name}")
    print()

    # Markdown word count
    print("MARKDOWN WORD COUNT")
    print("-" * 60)
    print(
        f"Total words: {total_markdown_words:,} / "
        f"{MAX_TOTAL_MARKDOWN_WORDS:,}"
    )

    if total_words_violation:
        print("VIOLATION: Total Markdown words exceed the 6,000-word limit.")
    else:
        print("OK: Total Markdown words are within the limit.")

    print()

    print("MARKDOWN CELL LIMIT")
    print("-" * 60)
    print(
        f"Markdown cells violating the "
        f"{MAX_WORDS_PER_MARKDOWN}-word limit: "
        f"{len(markdown_violations)}"
    )

    if markdown_violations:
        for violation in markdown_violations:
            print(
                f"  Cell {violation['cell']} "
                f"(Markdown #{violation['markdown_number']}): "
                f"{violation['words']} words"
            )
    else:
        print("OK: No Markdown cell exceeds 100 words.")

    print()

    print("CODE BLOCK LINE LIMIT")
    print("-" * 60)
    print(
        f"Code blocks violating the "
        f"{MAX_CODE_LINES}-line limit: "
        f"{len(code_line_violations)}"
    )

    if code_line_violations:
        for violation in code_line_violations:
            print(
                f"  Cell {violation['cell']} "
                f"(Code #{violation['code_number']}): "
                f"{violation['lines']} nonblank lines"
            )
    else:
        print("OK: No code block exceeds 30 nonblank lines.")

    print()

    print("FIGURE LIMIT")
    print("-" * 60)
    print(
        f"Total figures: {figure_count} / "
        f"{MAX_FIGURES}"
    )

    if figure_violation:
        print("VIOLATION: Number of figures exceeds the 15-figure limit.")
    else:
        print("OK: Number of figures is within the limit.")

    print()

    print("CODE COMMENT RULE")
    print("-" * 60)
    print(
        f"Code blocks containing comments: "
        f"{len(comment_violations)}"
    )

    if comment_violations:
        for violation in comment_violations:
            print(
                f"  Cell {violation['cell']} "
                f"(Code #{violation['code_number']})"
            )
    else:
        print("OK: No code blocks contain comments.")

    print()
    print("=" * 60)

    if has_violations:
        print("RESULT: VIOLATIONS FOUND")
        print("=" * 60)
    else:
        print("RESULT: NO VIOLATIONS")
        print("The notebook satisfies all specified restrictions.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Check restrictions for an IPython/Jupyter notebook."
    )

    parser.add_argument(
        "ipynb_file",
        help="Path to the .ipynb file"
    )

    args = parser.parse_args()

    ipynb_path = Path(args.ipynb_file)

    if not ipynb_path.exists():
        print(f"Error: File not found: {ipynb_path}")
        return

    if ipynb_path.suffix.lower() != ".ipynb":
        print("Error: Input file must be an .ipynb file.")
        return

    try:
        check_notebook(ipynb_path)
    except json.JSONDecodeError:
        print("Error: The notebook is not a valid JSON/IPYNB file.")
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()