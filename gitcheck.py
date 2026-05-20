import subprocess, os
out = []
for cmd in [
    ['git', 'log', '--oneline', '-3'],
    ['git', 'status', '--short'],
    ['git', 'remote', '-v'],
    ['git', 'diff', '--stat', 'HEAD'],
]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    out.append(f"$ {' '.join(cmd)}\n{r.stdout or '(no output)'}\n{r.stderr or ''}")

with open('gitcheck_result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
