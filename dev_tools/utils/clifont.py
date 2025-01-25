import re


class ReadOnlyMeta(type):
    def __setattr__(cls, key, value):
        raise AttributeError(f"Cannot modify read-only attribute {key}")


class CLIFont(metaclass=ReadOnlyMeta):
    '''Static class for formatting print statements on the terminal'''
    purple = "\033[95m"
    blue = "\033[94m"
    light_gray = "\033[90m"
    light_green = "\033[92m"
    bold = "\033[1m"
    reset = "\033[0m"


def print_bold(message, end="\n", flush=False):
    '''prints a message in bold'''
    print(f"{CLIFont.bold}{message}{CLIFont.reset}", end=end, flush=flush)


def print_insight(message, end="\n", flush=False):
    '''prints an insight from an agent'''
    print(f"{CLIFont.light_green}{message}{
          CLIFont.reset}", end=end, flush=flush)


def input_bold(message):
    '''wrapper for the input() function, showing prompt message in bold'''
    return input(f"{CLIFont.bold}{message}{CLIFont.reset}")


def print_cli_message(message, flush=False, end='\n'):
    # Replace bolded text (e.g. **text**) for the actual bold using shell fonts
    pattern = r"\*\*(.*?)\*\*"

    def repl(match):
        return f"{CLIFont.bold}{match.group(1)}{CLIFont.reset}"
    print(re.sub(pattern, repl, message), flush=flush, end=end)
