from __future__ import annotations

import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from main import MainApplication

PAGE_KEY = "recolor"
PAGE_MODULE = "modules.mod_recolor"


def main() -> None:
    if "--smoke-import-page" in sys.argv:
        __import__(PAGE_MODULE)
        return

    app = MainApplication()
    app.title("指定色替换")
    try:
        app.sidebar.pack_forget()
    except Exception:
        pass
    app.show_page(PAGE_KEY)
    app.mainloop()


if __name__ == "__main__":
    main()
