import argparse
import io
import logging
import os
import sys
import time
import warnings
import webbrowser
import functools
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

warnings.simplefilter("ignore", FutureWarning)

import argini
from tree_sitter import Node

from modules import grep
from modules import ts
from modules import helper

logging.basicConfig(format="%(message)s", level=logging.DEBUG)

IGNORE_ARGS: set[str] = set()


def iter_children_by_type(node: Node, type: str, recursive=True):
    if node.type == type:
        yield node
    elif node.children:
        for c in node.children:
            if recursive:
                yield from iter_children_by_type(c, type, recursive)
            elif node.type == type:
                yield node


def funcname_of_funcdef(node: Node | None) -> str:
    if node is None:
        return ""
    declarator = next(iter_children_by_type(node, "function_declarator"))
    identifier = declarator.child(0)
    return identifier.text.decode()


def is_static_func(node: Node | None) -> bool:
    if node is None:
        return False
    # if node.type == "function_declarator":
    #     prev = node.prev_sibling
    #     prev2 = prev.prev_sibling
    #     return b"static" in {prev.text, prev2.text}
    # else:
    static_marks = [n for n in node.children if n.type == "storage_class_specifier"]
    return static_marks != []


def lineno(lineno: int | str) -> str:
    return f"{lineno:>10}:"


def indent(ctx: bytes) -> bytes:
    return (b"\n" + lineno("").encode()).join(ctx.splitlines())


@dataclass
class CallSource:
    filepath: str
    lineno: int  # 1's based
    usage: Node
    callee: str
    argument: Node | None = None
    caller: Node = None
    conditions: list[Node] = field(default_factory=list)

    def _format_function_call(self) -> str:
        def _wrap_cond(conds: list[Node], pos: int, _depth=1) -> bytes:
            # for node in self.conditions:
            tab = _depth * b"    "
            if pos >= len(conds):
                return lineno(self.lineno).encode() + tab + b"%s"
            node = conds[pos]
            suffix = b""
            start_line = lineno(node.start_point[0]).encode()
            end_line = lineno(node.end_point[0]).encode()
            match node.type:
                case "if_statement":
                    if pos < len(conds) - 1 and conds[pos + 1].type == "else_clause":
                        # "if" here becase "else" clause
                        prefix = (
                            b"if %s {\n"
                            % indent(node.child_by_field_name("condition").text)
                            + lineno("").encode()
                            + tab
                            + b"}"
                        )
                        inner = _wrap_cond(conds, pos + 1, _depth)
                        suffix = b""
                    else:
                        prefix = b"if %s {" % indent(
                            node.child_by_field_name("condition").text
                        )
                        inner = _wrap_cond(conds, pos + 1, _depth + 1)
                        suffix = b"}"
                case "else_clause":
                    if pos < len(conds) - 1 and conds[pos + 1].type == "if_statement":
                        # "else" if
                        prefix = b"else "
                        inner = _wrap_cond(conds, pos + 1, _depth)
                        suffix = b""
                    else:
                        prefix = b"else {"
                        inner = _wrap_cond(conds, pos + 1, _depth + 1)
                        suffix = b"}"
                case "switch_statement":
                    prefix = b"switch %s {" % indent(
                        node.child_by_field_name("condition").text
                    )
                    inner = _wrap_cond(conds, pos + 1, _depth + 1)
                    suffix = b"}"
                case "case_statement":
                    val = node.child_by_field_name("value")
                    if val:
                        prefix = b"case %s: {" % indent(val.text)
                    else:
                        prefix = b"default: {"
                    suffix = b"}"
                    inner = _wrap_cond(conds, pos + 1, _depth + 1)
                case "for_statement":
                    types = [x.type for x in node.children]
                    left_paren = types.index("(")
                    right_paren = types.index(")")
                    prefix = b"for %s {" % b" ".join(
                        indent(x.text)
                        for x in node.children[left_paren : right_paren + 1]
                    )
                    inner = _wrap_cond(conds, pos + 1, _depth + 1)
                    suffix = b"}"
                case "do_statement":
                    prefix = b"do {"
                    suffix = b"} while %s;" % indent(
                        node.child_by_field_name("condition").text
                    )
                    inner = _wrap_cond(conds, pos + 1, _depth + 1)
                case "while_statement":
                    prefix = b"while %s {" % indent(
                        node.child_by_field_name("condition").text
                    )
                    inner = _wrap_cond(conds, pos + 1, _depth + 1)
                    suffix = b"}"
                case "labeled_statement":
                    tab = b""
                    prefix = node.child_by_field_name("label").text + b":"
                    inner = _wrap_cond(conds, pos + 1, _depth)
                    suffix = b" "
                case "compound_statement":
                    return _wrap_cond(conds, pos + 1, _depth)
                case _:
                    raise NotImplementedError(node)
            if suffix:
                return (
                    start_line
                    + tab
                    + prefix
                    + b"\n"
                    + inner
                    + b"\n"
                    + end_line
                    + tab
                    + suffix
                )
            else:
                return start_line + tab + prefix + b"\n" + inner

        combined_cond = _wrap_cond(list(reversed(self.conditions)), 0).decode()
        static = "static " if is_static_func(self.caller) else ""
        function_def = "unsl %s%s" % (static, funcname_of_funcdef(self.caller))
        passed_in_args, _ = self.get_caller_arguments()
        arg_txt = ("\n" + lineno("") + " " * (len(function_def) + 1)).join(
            ", ".join(passed_in_args[i : i + 5]) for i in range(0, len(passed_in_args), 5)
        )
        inner = combined_cond % indent(self.usage.text).decode()
        return "%s(%s) {\n%s\n%s}" % (
            lineno("") + function_def,
            arg_txt,
            inner,
            lineno(""),
        )

    def _format_table_init_declarator(self) -> str:
        def _pretty_indent(n, _depth=0):
            if n.parent:
                wrapper = _pretty_indent(n.parent, _depth + 1)
                if n.type == "initializer_list":
                    if _depth == 0:
                        end = "}" * len(wrapper)
                        return "%s\n    %s, ...\n%s" % (wrapper, n.text.decode(), end)
                    else:
                        return wrapper + "{"
                else:
                    return wrapper
            return ""

        declarator = self.caller
        while declarator.type != "declaration":
            declarator = declarator.parent
        type = declarator.child_by_field_name("type")
        varname = declarator.child_by_field_name("declarator").child_by_field_name(
            "declarator"
        )
        init_list = _pretty_indent(self.caller)
        return "{} {} = {};".format(
            type.text.decode(), varname.text.decode(), init_list
        )

    def get_caller_arguments(
        self, with_type: bool = True, only_related=False
    ) -> tuple[list[str], list[str]]:
        if self.caller.type in {
            # "ERROR",
            "function_definition",
            # "function_declarator",
            # "call_expression",
        }:
            args = [x
                for x in self.argument.children
                if x.type not in "(,)"
            ]
            args_ids = functools.reduce(lambda a, b: a + list(iter_children_by_type(b, "identifier")), args, [])
            args_id_txts = [x.text.decode() for x in args_ids]
            declarator = next(iter_children_by_type(self.caller, "function_declarator"))
            params = declarator.child_by_field_name("parameters")
            param_vars = {
                arg_name.text.decode(): arg_type.text.decode()
                for arg_name, arg_type in zip(
                    iter_children_by_type(params, "identifier"),
                    iter_children_by_type(params, "type_identifier"),
                )
            }
            local_arg_ids = list((set(args_id_txts) - set(param_vars.keys())) - IGNORE_ARGS)
            local_args = [x.text.decode() for (x, y) in zip(args, args_id_txts) if y in local_arg_ids]
            if only_related:
                pass_in_args = (set(param_vars.keys()) & set(args_id_txts)) - IGNORE_ARGS
                if with_type:
                    return [f"{param_vars.get(var, '')} {var}" for var in pass_in_args], local_args
                else:
                    return list(pass_in_args), local_args
            else:
                if with_type:
                    return [f"{param_vars[var]} {var}" for var in param_vars], local_args
                else:
                    return list(param_vars.keys()), local_args
        else:
            return [], []


# def _try_found_function(node: Node) -> Node:
#     _n = node.prev_sibling
#     while _n and _n.type != "function_declarator":
#         _n = _n.prev_sibling
#     assert _n is not None, "No function define found for %r" % node
#     return _n


def resolve_match(
    ts_node: Node, m: grep.Match, search_term: str
) -> tuple[Node, CallSource | None]:
    for node in ts_node.children:
        start_line, _ = node.start_point
        end_line, _ = node.end_point
        if m.lineno - 1 < start_line or end_line < m.lineno - 1:
            continue
        if search_term not in node.text.decode():
            continue
        if node.type in {
            "function_declarator",
            "comment",
            "identifier",
            "field_identifier",
            "string_literal",
        }:
            return node, None
        if node.type in {
            "declaration",
            "init_declarator",
            "preproc_if",
            "preproc_else",
            "preproc_ifdef",
            "expression_statement",
            "argument_list",
            "concatenated_string",
            "return_statement",
            "cast_expression",
            "pointer_expression",
            "type_definition",
            "struct_specifier",
            "field_expression",
            "field_declaration",
            "field_declaration_list",
        }:
            n, cs = resolve_match(node, m, search_term)
            if ts_node.parent is None:
                if cs and cs.caller is None:
                    raise NotImplementedError(node)
            #         cs.caller = _try_found_function(node)
            if cs:
                cs.usage = node
            return n, cs
        if node.type == "compound_statement":
            # { ... }
            n, cs = resolve_match(node, m, search_term)
            if cs and cs.conditions:
                cs.conditions.append(node)
            if ts_node.parent is None:
                if cs and cs.caller is None:
                    raise NotImplementedError(node)
            #         cs.caller = _try_found_function(node)
            return n, cs
        if node.type in {
            "if_statement",
            "else_clause",
            "switch_statement",
            "case_statement",
            "for_statement",
            "do_statement",
            "while_statement",
            "labeled_statement",
        }:
            n, cs = resolve_match(node, m, search_term)
            if cs is not None:
                cs.conditions.append(node)
            if ts_node.parent is None:
                if cs and cs.caller is None:
                    raise NotImplementedError(node)
                    # cs.caller = _try_found_function(node)
            return n, cs
        match node.type:
            # case "ERROR":
            #     n, cs = resolve_match(node, m, search_term)
            #     if cs is not None and cs.caller is None:
            #         cs.caller = node
            #     return n, cs
            # case "compound_statement":
            #     n, cs = resolve_match(node, m, search_term)
            #     if cs and cs.caller is None:
            #         prev = node.prev_sibling
            #         if prev and prev.type == "function_declarator":
            #             cs.caller = prev
            #     return n, cs
            case "initializer_list":
                n, cs = resolve_match(node, m, search_term)
                if n.type == "identifier":
                    if cs is None:
                        cs = CallSource(
                            filepath=m.filepath,
                            lineno=m.lineno,
                            usage=node,
                            callee=search_term,
                            caller=node,
                        )
                    return node, cs
                else:
                    if cs:
                        cs.usage = node
                    return n, cs
            case "preproc_def":
                n, _ = resolve_match(node, m, search_term)
                if n.type == "identifier":
                    return node, None
                else:
                    return n, None
            case "function_definition":
                n, cs = resolve_match(node, m, search_term)
                if n.type in {"call_expression", "assignment_expression"}:
                    if cs is not None:
                        if cs.caller is None:
                            cs.caller = node
                        elif cs.caller.type == "function_declarator":
                            cs.caller = node
                    return node, cs
                else:
                    return n, None
            case "assignment_expression":
                n, cs = resolve_match(node, m, search_term)
                if n.type == "identifier":
                    left = node.child_by_field_name("left")
                    assert left is not None
                    cs = CallSource(
                        filepath=m.filepath,
                        lineno=m.lineno,
                        usage=node,
                        callee=search_term,
                        caller=left,
                    )
                    return node, cs
                else:
                    if cs:
                        cs.usage = node
                    return n, cs
            case "call_expression":
                n, cs = resolve_match(node, m, search_term)
                args = node.child_by_field_name("arguments")
                if n.type in {"identifier", "field_identifier"}:
                    cs = CallSource(
                        filepath=m.filepath,
                        lineno=m.lineno,
                        usage=node,
                        callee=node.children[0].text.decode(),
                        argument=args,
                    )
                    return node, cs
                else:
                    if cs:
                        cs.usage = node
                    return n, cs
        raise NotImplementedError(node)
    else:
        # if b"#if 0" in ts_node.text:
        #     return ts_node, None
        # raise ValueError("Current syntax tree does not contain search term: %r" % search_term)
        logging.warning(
            "Search result for %r is ignored due to pre-process define. %s"
            % (search_term, m)
        )
        return ts_node, None


def trace_callmap(folder: str, search_term: str, trace_args=True):
    jobid = int(time.time())
    logging.info(f"[{jobid}] Start Search {search_term!r} in {folder!r}")
    logging.info(f"Option: {trace_args=}")
    logging.info(f"Option: {IGNORE_ARGS=}")

    call_maps: list[CallSource] = []

    search_queries: list[tuple[str, list[str]]] = [(search_term, [])]
    found = set()
    while search_queries:
        term, flists = search_queries.pop(0)
        if term in found:
            continue
        matches = grep.ripgrep(folder, term, flists)
        sorted_matches = sorted(matches, key=lambda m: "{}:{:>010}".format(m.filepath, m.lineno))
        found.add(term)

        logging.info("Search: %s" % term)
        for m in sorted_matches:
            filepath = Path(m.filepath)
            decoded_src = helper.byte2str(filepath.read_bytes())
            clean_src = ts.remove_if_0(decoded_src, m.filepath)
            root_node = ts.parse_syntax_tree(clean_src)

            _, cs = resolve_match(root_node, m, term)
            if cs:
                logging.debug(f"-- {m.relpath}:{m.lineno}")
                call_maps.append(cs)

                if cs.caller.type == "field_expression":
                    caller_funcname = list(
                        iter_children_by_type(cs.caller, "field_identifier")
                    )[-1]
                    logging.debug(">> %s = %s" % (caller_funcname, cs.callee))
                    next_term, staticfiles = caller_funcname, []
                elif cs.caller.type in {
                    # "ERROR",
                    "function_definition",
                    # "function_declarator",
                    # "call_expression",
                }:
                    caller_funcname = funcname_of_funcdef(cs.caller)
                    logging.debug(cs._format_function_call())
                    if is_static_func(cs.caller):
                        next_term, staticfiles = (caller_funcname, [cs.filepath])
                    else:
                        next_term, staticfiles = (caller_funcname, [])
                elif cs.caller.type == "initializer_list":
                    logging.debug(cs._format_table_init_declarator())
                    next_term, staticfiles = (None, [])
                else:
                    raise NotImplementedError(cs.caller)

                if trace_args and cs.caller.type in {
                    # "ERROR",
                    "function_definition",
                    # "function_declarator",
                    # "call_expression",
                }:
                    passed_in_args, _ = cs.get_caller_arguments(only_related=True)
                    if not passed_in_args:
                        continue

                if next_term:
                    rel_files = [os.path.relpath(p, folder) for p in staticfiles]
                    search_queries.append((next_term, rel_files))

    logging.info(f"[{jobid}] End")
    return call_maps


def merge_code_insert(curr: str, insert: str) -> str:
    """
    curr = AAA{
             BBBB
           }
    insert = AAA{
               DDDD
             }
    return: AAA{
              DDDD
              BBBB
            }
    """
    if curr == "":
        return insert

    olds = list(curr.splitlines())
    news = list(insert.splitlines())

    head = -1
    for head, (old, new) in enumerate(zip(olds, news)):
        if old != new:
            break
    assert head >= 0

    lbrace_cnt = Counter("".join(olds[:head]))["{"]
    rbrace_cnt = Counter("".join(olds[:head]))["}"]
    brace_cnt = lbrace_cnt - rbrace_cnt
    outs = olds[:head] + news[head:-brace_cnt] + olds[head:]
    return "\n".join(outs)


def write_pseudo_code(folder: str, sources: list[CallSource], fp=None):
    fp = fp or sys.stderr

    # merge caller
    func_calls: dict[Node, list[CallSource]] = defaultdict(list)
    for cs in sources:
        if cs.caller.type == "function_definition":
            func_calls[cs.caller].append(cs)

    func_printed = set()
    for cs in sources:
        if cs.caller.type == "field_expression":
            print(f"//-- {os.path.relpath(cs.filepath, folder)}:{cs.lineno}", file=fp)
            caller_funcname = list(
                iter_children_by_type(cs.caller, "field_identifier")
            )[-1]
            print(">> %s = %s" % (caller_funcname, cs.callee), file=fp)
        elif cs.caller.type in {
            # "ERROR",
            "function_definition",
            # "function_declarator",
            # "call_expression",
        }:
            if cs.caller not in func_printed:
                func_printed.add(cs.caller)
                cs_in_funcs = sorted(func_calls[cs.caller], key=lambda x: x.lineno)
                for adjacent_cs in cs_in_funcs:
                    print(
                        f"//-- {os.path.relpath(adjacent_cs.filepath, folder)}:{adjacent_cs.lineno}",
                        file=fp,
                    )
                func_txt = ""
                for adjacent_cs in reversed(cs_in_funcs):
                    func_txt = merge_code_insert(
                        func_txt, adjacent_cs._format_function_call()
                    )
                print(func_txt, file=fp)
        elif cs.caller.type == "initializer_list":
            print(f"//-- {os.path.relpath(cs.filepath, folder)}:{cs.lineno}", file=fp)
            print(cs._format_table_init_declarator(), file=fp)
        else:
            raise NotImplementedError(cs.caller)


def write_as_mermaid_flowchart(folder: str, sources: list[CallSource], fp, show_all_args = False):
    fn_filemaps: dict[str, str] = {}

    print("flowchart LR", file=fp)

    call_maps = [
        s
        for s in sources
        if s.caller.type
        in {
            # "ERROR",
            "function_definition",
            # "function_declarator",
            # "call_expression",
        }
    ]
    for src in call_maps:
        caller_funcname = funcname_of_funcdef(src.caller)
        fn_filemaps[caller_funcname] = src.filepath

    static_calls = defaultdict(list)
    public_calls = defaultdict(list)

    for src in call_maps:
        caller_funcname = funcname_of_funcdef(src.caller)
        caller_file = fn_filemaps.get(caller_funcname, "")
        callee_file = fn_filemaps.get(src.callee, "")
        passed_in_args, local_args = src.get_caller_arguments(with_type=False, only_related=True)
        if show_all_args:
            passed_args = ", ".join(passed_in_args + local_args)
        else:
            passed_args = ", ".join(passed_in_args)
        arg_txt = f"-- {passed_args} " if passed_args else ""
        if caller_file == callee_file:
            static_calls[caller_file].append((caller_funcname, src.callee, arg_txt))
        else:
            public_calls[src.filepath].append((caller_funcname, src.callee, arg_txt))

            # relpath = os.path.relpath(src.filepath, folder)
            # pathtag = relpath.replace("\\", "-")
            # print(f"subgraph {pathtag}", file=fp)
            # print(f"  {caller_funcname}", file=fp)
            # print("end", file=fp)
    def get_pathtag(abs_path: str) -> str:
        relpath = os.path.relpath(abs_path, folder)
        # sanitize invalid chars
        trans = str.maketrans("\\[]", "---")
        return relpath.translate(trans)

    for file, srcs in public_calls.items():
        callers = (x[0] for x in dict.fromkeys(srcs))
        pathtag = get_pathtag(file)
        print(f"subgraph {pathtag}", file=fp)
        for caller in dict.fromkeys(callers):
            print(f"  {caller}", file=fp)
        print("end", file=fp)

    for file, srcs in static_calls.items():
        pathtag = get_pathtag(file)
        print(f"subgraph {pathtag}", file=fp)
        for a, b, c in dict.fromkeys(srcs):
            print(f"  {a} {c}--> {b}", file=fp)
        print("end", file=fp)

    for file, srcs in public_calls.items():
        callers = (x[0] for x in srcs)
        for a, b, c in dict.fromkeys(srcs):
            print(f"{a} {c}--> {b}", file=fp)


def main(search_folder: str, search_term: str, trace_args = True, show_all_args = False):
    data_folder = helper.get_executable_folder(__file__) / "data"
    data_folder.mkdir(exist_ok=True)

    logfile = logging.FileHandler(data_folder / f"{search_term}.log")
    logfile.setLevel(logging.INFO)
    logfile.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(name)-5s][%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(logfile)
    sources = trace_callmap(search_folder, search_term, trace_args)

    pseudo_code_buf = io.StringIO()
    write_pseudo_code(search_folder, sources, pseudo_code_buf)

    diagram_buf = io.StringIO()
    write_as_mermaid_flowchart(search_folder, sources, diagram_buf, show_all_args)

    out1 = data_folder / f"{search_term}-pseudo_code.c"
    logging.info("PseudoCode: %s" % out1.absolute())
    with open(out1, "w") as fp:
        fp.write(pseudo_code_buf.getvalue())

    template = (Path(__file__).parent / "assets/index.html").read_text()
    out2 = data_folder / f"{search_term}.html"
    logging.info("FlowChart: %s" % out2.absolute())
    with open(out2, "w") as fp:
        webpage = template.replace(
            "<title>Document</title>",
            "<title>{} | {}</title>".format(search_term, search_folder),
        )
        webpage = webpage.replace("<diagram/>", diagram_buf.getvalue())
        webpage = webpage.replace("<source/>", pseudo_code_buf.getvalue())
        fp.write(webpage)

    webbrowser.open(str(out2))


class ValidFile(argini.Validator):
    @staticmethod
    def validate_input(input: str | list) -> bool:
        if isinstance(input, list):
            return False
        input = input.strip()
        return Path(input).exists() and Path(input).is_dir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_folder",
        help="Folder you want to grep.",
        required=True,
    )
    parser.add_argument(
        "--search_term",
        help="Function name you want to traceback",
        required=True,
    )
    parser.add_argument(
        "--trace_args",
        help="Set to True to stop recursively search when all passing arguments are local variables",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--show_all_args",
        help="Show all passing arguments in flowchart",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ignore_args",
        help="Ignore these argument names when tracing function arguments.",
        nargs="*",
        default=list(IGNORE_ARGS),
    )
    if len(sys.argv) == 1:
        import argini

        ini_folder = helper.get_executable_folder(__file__) / "ini"
        ini_folder.mkdir(exist_ok=True)
        ini = ini_folder / Path(__file__).with_suffix(".ini").name
        argini.import_from_ini(parser, ini)
        args = argini.get_user_inputs(parser)
        argini.save_to_ini(parser, ini, args)
    else:
        args = parser.parse_args()


    IGNORE_ARGS = set(args.ignore_args)

    try:
        main(args.search_folder, args.search_term, args.trace_args, args.show_all_args)
    except Exception:
        logging.fatal(__file__, exc_info=True)
        input("Press Enter to quit...")
