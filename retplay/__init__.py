"""RET Play GUI launcher."""
from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter
from typing import TYPE_CHECKING

from retplay.app import create_app
import os
from threading import Timer
import webbrowser

if TYPE_CHECKING:
    from typing import Optional, Sequence


def open_browser():
    """Opens browser tab."""
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:8050/")


def run_gui(cli_args: Optional[Sequence[str]] = None):
    """Return an instance of the RET Play gui.

    Args:
        cli_args (Optional[Sequence[str]]): Command line arguments. Defaults to None.
    """
    parser = ArgumentParser(description="Ret Play", formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "playback",
        nargs="?",
        type=str,
        help="(optional) RET Playback directory."
        + " This must be a folder containing a file called 'playback.json'."
        + "\n   A map file should be called 'base_map.png'"
        + " and located in a sub-folder called 'assets'. "
        + "\n   Any icon files should be of filetype .svg"
        + " and located in a sub-folder of 'assets' called 'icons'. ",
    )

    args = parser.parse_args(cli_args)

    app = create_app(args.playback)

    print("App created, opening server...")
    Timer(1, open_browser).start()
    app.run_server(debug=True)


if __name__ == "__main__":
    run_gui()
