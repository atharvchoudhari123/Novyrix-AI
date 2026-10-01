from __future__ import annotations

import ast
import operator

_ALLOWED = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def calculate(expression: str):
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("Expression is empty.")
    if len(expression) > 300:
        raise ValueError("Expression is too long.")
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp):
            fn = _ALLOWED.get(type(node.op))
            if fn is None:
                raise ValueError("Unsupported operator.")
            return fn(visit(node.operand))
        if isinstance(node, ast.BinOp):
            fn = _ALLOWED.get(type(node.op))
            if fn is None:
                raise ValueError("Unsupported operator.")
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 20:
                raise ValueError("Exponent is too large.")
            return fn(left, right)
        raise ValueError("Unsupported expression.")
    return visit(tree)
