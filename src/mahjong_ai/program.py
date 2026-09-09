"""A bounded scoring-expression language for generated rule changes.

Expressions can branch on public policy features but cannot import, access
files, mutate engine state, or execute arbitrary statements.
"""
import ast
import math
from functools import lru_cache

NAMES = {'base', 'risk', 'progress', 'remaining', 'score', 'gap', 'fan', 'quality',
         'quality_delta', 'piao', 'discard', 'hu', 'kong', 'call', 'white',
         'hand', 'scores', 'state', 'rank', 'ranks', 'abs', 'min', 'max', 'log1p'}
NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
         ast.Name, ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
         ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Lt, ast.LtE, ast.Gt,
         ast.GtE, ast.Eq, ast.NotEq, ast.Call, ast.Subscript)


@lru_cache(maxsize=128)
def compile_program(source):
    if not isinstance(source, str) or len(source) > 2000:
        raise ValueError('rule expression must be <=2000 characters')
    tree = ast.parse(source, mode='eval')
    nodes = list(ast.walk(tree))
    if len(nodes) > 160:
        raise ValueError('rule expression too complex')
    for node in nodes:
        if not isinstance(node, NODES):
            raise ValueError(f'forbidden rule syntax: {type(node).__name__}')
        if isinstance(node, ast.Name) and node.id not in NAMES:
            raise ValueError(f'unknown rule feature: {node.id}')
        if isinstance(node, ast.Constant) and (not isinstance(node.value, (int, float, bool))
                                               or not math.isfinite(node.value) or abs(node.value) > 1e6):
            raise ValueError('invalid rule constant')
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                    or node.func.id not in {'abs', 'min', 'max', 'log1p'} or node.keywords):
            raise ValueError('only bounded numeric calls permitted')
        if isinstance(node, ast.Subscript) and (not isinstance(node.value, ast.Name)
                    or node.value.id not in {'hand', 'scores', 'state', 'ranks'}
                    or not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int)):
            raise ValueError('only constant indexes into public vectors permitted')
    return compile(tree, '<generated-rule>', 'eval')


def evaluate_program(source, features):
    value = eval(compile_program(source), {'__builtins__': {}},
                 dict(features, abs=abs, min=min, max=max, log1p=math.log1p))
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('rule expression returned a nonfinite/non-numeric score')
    return value
