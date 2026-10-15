TOOL_LEVELS = {
    # Advanced beta gives the lightweight tier the same safe tool surface.
    # The model still decides what to say; tools are only executed when the
    # deterministic router explicitly selects them.
    "novyrix-3.2": {
        "calculator", "list_files", "read_file", "write_file", "edit_file",
        "search_files", "analyze_traceback", "check_syntax", "run_tests",
        "web_search", "fetch_webpage",
    },
    "novyrix-4.0": {
        "calculator", "list_files", "read_file", "write_file", "edit_file",
        "search_files", "analyze_traceback", "check_syntax", "run_tests",
        "web_search", "fetch_webpage",
    },
    "novyrix-5.7": {
        "calculator", "list_files", "read_file", "write_file", "edit_file",
        "search_files", "analyze_traceback", "check_syntax", "run_tests",
        "web_search", "fetch_webpage",
    },
}


def allowed_tools(model_id: str):
    return TOOL_LEVELS.get(model_id, set())
