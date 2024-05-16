import os
from time import sleep

from loguru import logger
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.progress import Progress
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree


console = Console()

# Setup logging with rich handler
FORMAT = "%(time)s %(level)s %(message)s"
logger.remove()
logger.add(RichHandler(console=console, rich_tracebacks=True), format=FORMAT)

console = Console()
custom_style = Style(color="blue", bold=True)

with open("README.md") as readme:
    markdown = Markdown(readme.read(), style=custom_style)
console.print(markdown)


# logging_config.py
def log_tasks(tasks):
    with console.status("[bold green]Working on tasks...", spinner="dots2") as status:
        for task_name, task_func in tasks:
            sleep(1)
            console.log(f"{task_name} complete")


def setup_logger():
    logger.remove()
    # Add a new handler with the correct format for loguru
    logger.add(RichHandler(rich_tracebacks=True),
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{"
                      "name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    return logger


def log_table(data, title="Table", caption="", show_lines=True, show_footer=False, show_edge=True, highlight=True,
              row_styles=None, border_style="blue", title_style="bold red", caption_style="bold green"):
    if row_styles is None:
        row_styles = ["dim", ""]
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, title=title, caption=caption,
                  show_lines=show_lines, show_footer=show_footer, show_edge=show_edge, highlight=highlight,
                  row_styles=row_styles, border_style=border_style, title_style=title_style,
                  caption_style=caption_style)
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 100  # Set a default value for terminal_width
    for header in data[0].keys():
        table.add_column(header, justify="center", style="cyan", no_wrap=True, width=terminal_width)
    for row in data:
        table.add_row(*[str(value) for value in row.values()])
    console.print("📊", table)  # Emoji: Bar Chart


def log_progress(task_description, completed, total):
    with Progress() as progress:
        task = progress.add_task(task_description, total=total)
        while not progress.finished:
            progress.update(task, advance=completed)
    console.print("⏳", f"Progress: {task_description} - {completed}/{total}")  # Emoji: Hourglass


def log_status(status_message):
    console.print("✅", f"[bold green]{status_message}[/bold green]")  # Emoji: Check Mark


def log_columns(data):
    console.print("📚", data, justify="center")  # Emoji: Books


def log_markdown(data):
    markdowns = Markdown(data)
    console.print("📝", markdowns)  # Emoji: Memo


def log_error(message):
    logger.error(message)
    console.print(f"[bold red]Error:[/bold red] {message}")


def log_syntax_highlighting(code, language="python"):
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print("🖥️", syntax)  # Emoji: Desktop Computer


def log_tree(data, parent=None, indent=0):
    for key, value in data.items():
        if isinstance(value, dict):
            node = Tree(f"[cyan]{key}[/cyan]")
            log_tree(value, node, indent + 1)
            if parent is None:
                console.print("🌳", node)
            else:
                parent.add(node)
        else:
            if parent is None:
                console.print("🌳", f"[cyan]{key}[/cyan]")
            else:
                parent.add(f"[cyan]{key}[/cyan]")
