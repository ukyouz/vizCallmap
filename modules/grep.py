import base64
import csv
import os
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from modules import helper


class CommandError(Exception):
    """ fail to run grep command """


@dataclass
class Match:
    root: str
    relpath: str
    lineno: int  # 1's based
    text: str

    @property
    def filepath(self):
        return os.path.join(self.root, self.relpath)

    def __str__(self):
        return "{}:{}:{}".format(
            self.relpath,
            self.lineno,  # 1's based
            self.text,
        )


def ripgrep(root: str, search_term: str, filelists: list[str]=None) -> list[Match]:
    filelists = filelists or []
    rg_proc = subprocess.run(
        [
            "rg",
            "-w",
            "--json",
            "--no-encoding",
            "--glob", "*.c",
            "--glob", "*.h",
            "--glob", "*.hh",
            "--glob", "*.C",
            "--glob", "*.cpp",
            "--glob", "*.CPP",
            search_term,
            *filelists,
        ],
        capture_output=True,
        cwd=root,
    )

    if rg_proc.stderr:
        raise CommandError(rg_proc.stderr)

    outputs = rg_proc.stdout.decode().splitlines()
    jsons = [json.loads(o) for o in outputs]

    matches = []
    for j in jsons:
        if j["type"] != "match":
            continue
        data = j["data"]

        line_data = data["lines"]
        if "text" in line_data:
            line = line_data["text"].rstrip()
        elif "bytes" in line_data:
            # https://github.com/eclipse-theia/theia/issues/2622
            bline = base64.b64decode(line_data["bytes"])
            line = helper.byte2str(bline)
        else:
            import logging
            from pprint import pformat
            logging.error("filelists = \n" + pformat(filelists, indent=2))
            logging.error("data = \n" + pformat(data, indent=2))
            raise KeyError("Unknown json format for data['lines']")

        match = Match(
                root,
                data["path"]["text"],
                data["line_number"],
                line,
            )
        matches.append(match)
    return matches
