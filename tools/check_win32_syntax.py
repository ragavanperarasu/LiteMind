#!/usr/bin/env python3
"""
Compile the Windows-only code paths on a non-Windows machine.

LiteMind has two files with `#if defined(_WIN32)` branches: MappedFile.cpp,
which does the memory mapping, and main.cpp, which sets the console code page.
Those branches are invisible to a Linux or macOS build, so a mistake in them is
only found when someone builds on Windows -- which, if that someone is a
colleague on another continent, is a slow way to learn.

This compiles them here. It copies each source, rewrites `#if defined(_WIN32)`
to `#if 1` and `#if !defined(_WIN32)` to `#if 0`, and compiles the result
against the stub Win32 headers in tools/win32stub/. The guards are flipped
rather than _WIN32 defined, because defining _WIN32 makes the host's standard
library try to be the Windows one and fail long before it reaches this code.

The stub declares only what LiteMind calls and implements nothing, so this
checks that the code compiles, not that it runs. Compiling is the part that
cannot be checked any other way from here.

    python3 tools/check_win32_syntax.py

Exits non-zero if anything fails to compile. Skips itself with a clear message
when no suitable compiler is present.
"""

import os
import shutil
import subprocess
import sys
import tempfile

# Each source and the guards to flip in it.
SOURCES = [
    ("src/MappedFile.cpp", True),
    ("src/main.cpp", False),
]


def find_compiler():
    for name in ("clang++", "g++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stub = os.path.join(root, "tools", "win32stub")

    if sys.platform == "win32":
        print("Running on Windows; build normally instead.")
        return 0

    compiler = find_compiler()
    if not compiler:
        print("No C++ compiler found; skipping.", file=sys.stderr)
        return 0

    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        for relative, flip_negative in SOURCES:
            source_path = os.path.join(root, relative)
            with open(source_path, encoding="utf-8") as handle:
                text = handle.read()

            if "#if defined(_WIN32)" not in text:
                print(f"  {relative}: no Windows branch found, skipping")
                continue

            text = text.replace("#if defined(_WIN32)", "#if 1")
            if flip_negative:
                text = text.replace("#if !defined(_WIN32)", "#if 0")

            generated = os.path.join(scratch, os.path.basename(relative))
            with open(generated, "w", encoding="utf-8") as handle:
                handle.write(text)

            command = [
                compiler, "-std=c++20", "-fsyntax-only",
                "-Wall", "-Wextra", "-Wpedantic",
                "-I", stub,
                "-I", os.path.join(root, "include"),
                generated,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  {relative}: Windows branch compiles")
            else:
                failures += 1
                print(f"  {relative}: FAILED", file=sys.stderr)
                print(result.stderr, file=sys.stderr)

    if failures:
        print(f"\n{failures} file(s) failed.", file=sys.stderr)
        return 1
    print("\nWindows code paths compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
