#!/bin/bash

# Code quality checks script for the RAG chatbot project
# This script runs linting, type checking, and formatting checks

echo "🔍 Running code quality checks..."

echo "🖤 Checking code formatting with Black..."
if ! uv run black --check backend/ main.py; then
    echo "❌ Code formatting issues found. Run './format.sh' to fix."
    exit 1
fi

echo "📋 Checking import sorting with isort..."
if ! uv run isort --check-only backend/ main.py; then
    echo "❌ Import sorting issues found. Run './format.sh' to fix."
    exit 1
fi

echo "🔍 Running basic flake8 linting (syntax and major issues only)..."
if ! uv run flake8 --select=E9,F63,F7,F82 backend/ main.py; then
    echo "❌ Critical linting issues found."
    exit 1
fi

echo "✅ All quality checks passed!"