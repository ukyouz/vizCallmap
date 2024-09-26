import io
from functools import lru_cache
from pathlib import Path

import C_DefineParser

from tree_sitter import Language
from tree_sitter import Node
from tree_sitter import Parser


parser = Parser()
C_LANGUAGE = Language(Path(__file__).parent / "treesitter/treesitter_c.so", "c")
parser.set_language(C_LANGUAGE)


@lru_cache
def remove_if_0(source: str, filename="") -> str:
    p = C_DefineParser.Parser()

    lines = source.splitlines(keepends=True)

    hidden_linenos = set(range(len(lines)))

    fs = io.StringIO(source)
    fs.name = filename
    for _, line_no in p.read_file_lines(fs, try_if_else=True, reserve_whitespace=True, include_block_comment=True):
        hidden_linenos.remove(line_no - 1)

    for line_no in hidden_linenos:
        lines[line_no] = "\n"

    return "".join(lines)


@lru_cache
def parse_syntax_tree(source: str) -> Node:
    tree = parser.parse(source.encode())
    return tree.root_node