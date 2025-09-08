from typing import List, Union, Literal, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field, validator

app = FastAPI(title="Calculator API", version="1.0.0")

CURRENT_EXPRESSION: Optional[str] = None

class BinaryOp(str):
    pass

AllowedOp = Literal["+", "-", "*", "/"]

class ABOpRequest(BaseModel):
    a: Union[float, str]
    op: AllowedOp
    b: Union[float, str]
    combine_with_current: Literal["replace", "append_left", "append_right"] = "replace"

    @validator("a", "b")
    def validate_operand(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Пустая строка недопустима")
        return v

class ExprStringRequest(BaseModel):
    expr: str
    variables: Optional[dict] = None

class ExecResponse(BaseModel):
    expression: str
    rpn: List[str]
    result: float

OPS_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
OPS_FUNC = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
}

def tokenize(expr: str) -> List[str]:
    s = expr.replace(" ", "")
    tokens: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isdigit() or (ch == "." and i+1 < len(s) and s[i+1].isdigit()):
            j = i+1
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i+1
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        if ch in "+-*/()":
            if ch == "-":
                prev = tokens[-1] if tokens else None
                if prev is None or prev in OPS_PRECEDENCE or prev == "(":
                    tokens.extend(["0", "-"])
                    i += 1
                    continue
            tokens.append(ch)
            i += 1
            continue
        raise HTTPException(status_code=400, detail=f"Недопустимый символ: {ch}")
    return tokens

def to_rpn(tokens: List[str]) -> List[str]:
    output, stack = [], []
    for tok in tokens:
        if tok.replace(".", "", 1).isdigit() or is_identifier(tok):
            output.append(tok)
        elif tok in OPS_PRECEDENCE:
            while stack and stack[-1] in OPS_PRECEDENCE and OPS_PRECEDENCE[stack[-1]] >= OPS_PRECEDENCE[tok]:
                output.append(stack.pop())
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise HTTPException(status_code=400, detail="Несбалансированные скобки")
            stack.pop()
        else:
            raise HTTPException(status_code=400, detail=f"Неожиданный токен: {tok}")
    while stack:
        top = stack.pop()
        if top in ("(", ")"):
            raise HTTPException(status_code=400, detail="Несбалансированные скобки")
        output.append(top)
    return output

def is_identifier(tok: str) -> bool:
    if not tok:
        return False
    if tok[0].isalpha() or tok[0] == "_":
        return all(c.isalnum() or c == "_" for c in tok[1:])
    return False

def eval_rpn(rpn: List[str], variables: Optional[dict] = None) -> float:
    stack: List[float] = []
    for tok in rpn:
        if tok in OPS_FUNC:
            if len(stack) < 2:
                raise HTTPException(status_code=400, detail="Недостаточно операндов")
            b = stack.pop()
            a = stack.pop()
            if tok == "/" and b == 0:
                raise HTTPException(status_code=400, detail="Деление на ноль")
            stack.append(OPS_FUNC[tok](a, b))
        else:
            if tok.replace(".", "", 1).isdigit():
                stack.append(float(tok))
            elif is_identifier(tok):
                if not variables or tok not in variables:
                    raise HTTPException(status_code=400, detail=f"Неизвестная переменная: {tok}")
                stack.append(float(variables[tok]))
            else:
                raise HTTPException(status_code=400, detail=f"Неожиданный токен: {tok}")
    if len(stack) != 1:
        raise HTTPException(status_code=400, detail="Ошибка вычисления")
    return stack[0]

def normalize_operand(x: Union[float, str]) -> str:
    return str(x).strip()

def combine_expressions(current_expr, a, op, b, mode):
    left = normalize_operand(a)
    right = normalize_operand(b)
    piece = f"({left}{op}{right})"
    if mode == "replace" or not current_expr:
        return piece
    if mode == "append_left":
        return f"({current_expr}{op}{piece})"
    if mode == "append_right":
        return f"({piece}{op}{current_expr})"
    return piece

@app.get("/add")
def add(x: float, y: float):
    return {"result": x + y, "expression": f"({x}+{y})"}

@app.get("/subtract")
def subtract(x: float, y: float):
    return {"result": x - y, "expression": f"({x}-{y})"}

@app.get("/multiply")
def multiply(x: float, y: float):
    return {"result": x * y, "expression": f"({x}*{y})"}

@app.get("/divide")
def divide(x: float, y: float):
    if y == 0:
        raise HTTPException(status_code=400, detail="Деление на ноль")
    return {"result": x / y, "expression": f"({x}/{y})"}

@app.get("/expression")
def get_current_expression():
    global CURRENT_EXPRESSION
    return {"expression": CURRENT_EXPRESSION}

@app.post("/expression/set", response_model=ExecResponse)
def set_expression(payload: ExprStringRequest):
    global CURRENT_EXPRESSION
    expr = payload.expr.strip()
    tokens = tokenize(expr)
    rpn = to_rpn(tokens)
    result = eval_rpn(rpn, payload.variables)
    CURRENT_EXPRESSION = expr
    return ExecResponse(expression=expr, rpn=rpn, result=result)

@app.post("/expression/append")
def append_piece(payload: ABOpRequest):
    global CURRENT_EXPRESSION
    new_expr = combine_expressions(CURRENT_EXPRESSION, payload.a, payload.op, payload.b, payload.combine_with_current)
    tokens = tokenize(new_expr)
    rpn = to_rpn(tokens)
    CURRENT_EXPRESSION = new_expr
    return {"expression": CURRENT_EXPRESSION, "rpn": rpn}

@app.post("/expression/execute", response_model=ExecResponse)
def execute_expression(variables: Optional[dict] = None):
    global CURRENT_EXPRESSION
    if not CURRENT_EXPRESSION:
        raise HTTPException(status_code=400, detail="Нет выражения")
    rpn = to_rpn(tokenize(CURRENT_EXPRESSION))
    result = eval_rpn(rpn, variables)
    return ExecResponse(expression=CURRENT_EXPRESSION, rpn=rpn, result=result)
