
import re, pathlib
from collections import Counter

SAFE = set("""Delta Rightarrow approx ast beta boxed cdot dots equiv exp frac ge hat in inf
infty int langle le left ln log lim longmapsto longrightarrow mapsto mathbb mathbf mathcal
mathrm max mid min mu ne partial pi qquad quad rangle right rightarrow sigma sqrt sum sup
tau text tilde to top widehat xi ldots""".split())

MD_ACTIVE = [(re.compile(r"^\s*=+\s*$"), "SETEXT-H1"),
             (re.compile(r"^\s*-{1,}\s*$"), "SETEXT-H2/HR"),
             (re.compile(r"^\s*[-*+]\s"), "LIST"),
             (re.compile(r"^\s*#{1,6}\s"), "HEADING"),
             (re.compile(r"^\s*>"), "BLOCKQUOTE"),
             (re.compile(r"^\s*\d+\.\s"), "ORDERED-LIST"),
             (re.compile(r"^\s*\|"), "TABLE"),
             (re.compile(r"^\s*_{3,}\s*$"), "HR")]
NONASCII = re.compile(r"[^\x00-\x7F]")

issues=[]
def add(f,ln,k,t): issues.append((f,ln,k,str(t).strip()[:90]))

def check_body(f, ln, body):
    if NONASCII.sub("", body) != body:
        add(f, ln, "NON-ASCII-IN-MATH", "".join(sorted(set(NONASCII.findall(body)))) + " :: " + body)
    for c in set(re.findall(r"\\([A-Za-z]+)", body)):
        if c not in SAFE: add(f, ln, "UNSAFE-CMD", "\\" + c)
    d = 0
    for m in re.finditer(r"(?<!\\)\{|(?<!\\)\}", body):
        d += 1 if m.group() == "{" else -1
        if d < 0: add(f, ln, "UNBALANCED-BRACE", body); return
    if d: add(f, ln, "UNBALANCED-BRACE", body)
    for m in re.finditer(r"[\^_]", body):
        n = body[m.end():m.end()+1]
        if n == "" or n in " })],;": add(f, ln, "DANGLING-SUB/SUP", body[max(0,m.start()-25):m.start()+25])
    if "*" in body: add(f, ln, "ASTERISK-IN-MATH", body)
    if ";" in body: add(f, ln, "SEMICOLON-IN-MATH", body)

for p in sorted(pathlib.Path(".").glob("*.md")):
    lines = p.read_text(encoding="utf-8").split("\n")
    in_c = in_d = False; st = 0; buf = []
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if s.startswith("```"): in_c = not in_c; continue
        if in_c: continue
        if s == "$$":
            if not in_d:
                in_d = True; st = i; buf = []
                if l != "$$": add(p.name, i, "INDENTED-FENCE", l)
                if i > 1 and lines[i-2].strip() != "": add(p.name, i, "NO-BLANK-BEFORE", lines[i-2])
            else:
                in_d = False; check_body(p.name, st, " ".join(buf))
                if i < len(lines) and lines[i].strip() != "": add(p.name, i, "NO-BLANK-AFTER", lines[i])
            continue
        if in_d:
            buf.append(l)
            if s == "": add(p.name, i, "BLANK-IN-BLOCK", "(blank)")
            for rx, name in MD_ACTIVE:
                if rx.match(l): add(p.name, i, "MD-ACTIVE-LINE:" + name, l)
            continue
        if s.startswith(">") and "$$" in s: add(p.name, i, "DISPLAY-IN-BLOCKQUOTE", l)
        if s.startswith("#") and "$" in s: add(p.name, i, "MATH-IN-HEADING", l)
        if "$$" in s and s != "$$": add(p.name, i, "INLINE-DOUBLE-DOLLAR", l)
        for m in re.finditer(r"(?<!\$)\$([^$\n]+)\$(?!\$)", l):
            b = m.group(1)
            check_body(p.name, i, b)
            if b[0].isspace() or b[-1].isspace(): add(p.name, i, "SPACE-AT-DELIM", b)
            if l.lstrip().startswith("|") and "|" in b: add(p.name, i, "PIPE-IN-TABLE-MATH", b)
            bef = l[m.start()-1] if m.start() else " "
            aft = l[m.end()] if m.end() < len(l) else " "
            if bef.isalnum() or aft.isalnum(): add(p.name, i, "GLUED-INLINE-MATH", b)
        if len(re.findall(r"(?<!\\)\$", l)) % 2 and "$$" not in l: add(p.name, i, "ODD-DOLLAR", l)
    if in_d: add(p.name, st, "UNCLOSED-BLOCK", "$$")

for x in issues: print("{}:{}: {}: {}".format(*x))
print("\\nTOTAL: %d" % len(issues))
for k, c in Counter(k for _,_,k,_ in issues).most_common(): print("  %s: %d" % (k, c))

