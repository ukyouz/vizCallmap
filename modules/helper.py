import os
import sys
from pathlib import Path

def get_executable_folder(py_main_file_path: str) -> Path:
    # determine if application is a script file or frozen exe
    if getattr(sys, "frozen", False):
        application_path = os.path.dirname(sys.executable)
    elif py_main_file_path:
        application_path = os.path.dirname(py_main_file_path)
    return Path(application_path)


def byte2str(bstr: bytes) -> str:
    line = repr(bstr)  # fallback when all encoding method fails
    for encoding in ("cp932", "sjis", "euc-jp", "utf-8"):
        try:
            line = bstr.decode(encoding)
        except UnicodeDecodeError:
            continue
    return line