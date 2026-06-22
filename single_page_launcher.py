from __future__ import annotations

import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from main import MainApplication


def main() -> None:
    app = MainApplication()
    app.title("指定色替换")
    try:
        app.sidebar.pack_forget()
    except Exception:
        pass
    app.show_page("recolor")
    app.mainloop()


if __name__ == "__main__":
    main()
