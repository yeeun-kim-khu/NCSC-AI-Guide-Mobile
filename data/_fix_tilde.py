import os, glob

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

for f in files:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with open(f, "r", encoding=enc) as fp:
                content = fp.read()
            if "~" in content:
                with open(f, "w", encoding=enc) as fp:
                    fp.write(content.replace("~", "-"))
                print(f"Fixed: {os.path.relpath(f, data_dir)}")
            break
        except Exception:
            continue

print("Done.")
