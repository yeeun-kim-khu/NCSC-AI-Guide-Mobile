import subprocess, sys, os

cwd = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    print(f"$ {' '.join(cmd)}")
    print(r.stdout.strip() or "(no stdout)")
    if r.stderr.strip():
        print("STDERR:", r.stderr.strip())
    print("RC:", r.returncode)
    print()
    return r.returncode

run(['git', 'add', '-A'])
rc = run(['git', 'commit', '-m', 'fix: clarification 제거, FAQ 라우팅 수정 (2026-05-21)'])
if rc == 0:
    run(['git', 'push'])
else:
    print("Nothing new to commit or commit failed. Trying push anyway...")
    run(['git', 'push'])
