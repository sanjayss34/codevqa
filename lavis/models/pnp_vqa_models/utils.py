from copy import deepcopy
import ast

def anonymize_ast(node, protected_names):
    new_node = deepcopy(node)
    if isinstance(new_node, ast.Name):
        if new_node.id not in protected_names:
            new_node.id = "var"
    elif isinstance(new_node, ast.Constant):
        new_node.value = "value"
    elif isinstance(new_node, ast.AST):
        for k, v in vars(node).items():
            if k in {'parent', 'func', 'lineno', 'end_lineno', 'col_offset', 'end_col_offset', 'ctx'}:
                continue
            setattr(new_node, k, anonymize_ast(v, protected_names))
    elif isinstance(new_node, list):
        new_node = [anonymize_ast(child, protected_names) for child in new_node]
    return new_node

def collect_functions(node):
    if hasattr(node, "func") and hasattr(node.func, "id"):
        return {node.func.id}
    func_names = set()
    if isinstance(node, ast.AST):
        for k, v in vars(node).items():
            if k in {'parent', 'func', 'lineno', 'end_lineno', 'col_offset', 'end_col_offset', 'ctx'}:
                continue
            func_names = func_names.union(collect_functions(v))
    elif isinstance(node, list):
        for child in node:
            func_names = func_names.union(collect_functions(child))
    return func_names
