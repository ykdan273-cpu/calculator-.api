# Calculator API (FastAPI)

Простое FastAPI-приложение, реализующее калькулятор.

## Возможности
- Простые операции: сложение, вычитание, умножение, деление
- Добавление выражений по шагам (a, op, b)
- Установка выражения строкой `(a+b)*c + (d-e)/(f-g)`
- Просмотр текущего выражения
- Выполнение выражения с поддержкой переменных

## Установка и запуск

```bash
git clone https://github.com/<ВАШ_НИК>/calculator-api.git
cd calculator-api
pip install -r requirements.txt
uvicorn main:app --reload
```

Документация будет доступна по адресу:
- Swagger UI: http://127.0.0.1:8000/docs
