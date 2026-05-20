@echo off
git log --oneline -3
echo ---GIT STATUS---
git status --short
echo ---REMOTE---
git remote -v
