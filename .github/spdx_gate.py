#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every source file declares the workspace licence, and keeps declaring it.

    python3 .github/spdx_gate.py            # check
    python3 .github/spdx_gate.py --self-test

A SWEEP DOES NOT FIX A PROPERTY THAT REGROWS. `fuel` stamped 795 files on
2026-08-19 and had drifted to 823/833 by 2026-09-10 -- not because anything was
undone, but because ten files were added afterwards and nothing was watching.
The sweep is the one-off; this is the part that lasts.

WHAT THIS REFUSES TO DO, AND WHY EACH ONE IS DELIBERATE:

  * It never WRITES. A gate that repairs the thing it measures reports success
    forever and tells you nothing about whether anyone is maintaining it.

  * It never treats a DIFFERENT identifier as a failure to be corrected.
    `fuel-examples/src/bs1770.rs` is a verbatim Apache-2.0-ONLY third-party
    work, and fuel's blanket sweep stamped `MIT OR Apache-2.0` onto it --
    asserting a licence grant nobody made. Files whose licence is somebody
    else's decision belong in HOLDOUT below, by name, with a reason.

  * It refuses to pass on an EMPTY population. `0 of 0 files` is 100% compliant
    by arithmetic and means the glob matched nothing -- a renamed directory, a
    moved workspace, a typo in an extension. The comforting number and the
    broken query are the same number, and nobody audits good news.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell, no caller input
import sys

LICENCE = "MIT OR Apache-2.0"
# ⚠️ AN EXTENSION THIS TUPLE OMITS IS A POPULATION THE GATE NEVER COUNTED, and
# the omission shows up as a CLEANER number rather than a smaller one. Measured
# elsewhere in this portfolio: `fuel` is booked at "833/833, 100%" and that is
# 833 of 833 `.rs` - its 16 `.metal` files were never in the denominator, and
# three of them carry `Copyright (c) 2024 Apple Inc.`
#
# This workspace is `.rs` only, plus this script. Verified against `git ls-files`
# rather than assumed: 39 .rs, 9 .md, 6 .toml, 1 .yml. If that ever stops being
# true, the file that changes it is the file that must extend this tuple.
EXTENSIONS = (".rs", ".py")

#: How far into a file the header may sit. Measured, not guessed: across 2,100
#: real .rs files in this portfolio (fuel, kiss-ref, lightbulb, synapse), 851
#: SPDX lines sit at line index 0, one at index 4, and NONE at index >= 10.
HEADER_WINDOW = 10

#: The gate's own floor. If the tree ever holds fewer source files than this,
#: the query is broken rather than the tree suddenly minimal. Set below the real
#: count (40) on purpose: this catches a glob that matches nothing or almost
#: nothing, not ordinary deletion.
MINIMUM_FILES = 30

#: Paths this gate must NOT require a header on, each with the reason it is here.
#: Empty, and that is now CHECKED rather than asserted - see `survey_copyright`.
#:
#: 🔴 THIS COMMENT USED TO CARRY A COUNT, AND THE COUNT WAS WRONG TWICE.
#: #43 stated `-> 0` and a control of `-> 8`. #44 corrected the control to 9 by
#: ADDING ONE for the gate script rather than re-running the query; the true
#: figure was 10, because `CHANGELOG.md` began matching in #42 four days
#: earlier. Then #44 asserted the inherited 8 had been "true at that instant",
#: which it never was.
#:
#: ⚠️ A CORRECTION DERIVED BY ARITHMETIC ON A STALE BASE INHERITS THE STALENESS
#: IT IS CORRECTING - and vouching for the inherited number made it HARDER to
#: catch, not easier. A wrong number invites re-derivation; a wrong number with
#: a provenance claim attached discourages it.
#:
#: ⚠️ THE COUNT DRIFTED TWICE IN FOUR DAYS, from two unrelated PRs, with neither
#: author doing anything wrong. The next drift is certain and only its date is
#: unknown. A COUNT CANNOT SURVIVE TREE GROWTH; A PROPERTY CAN - so the number
#: is gone and the property it was standing in for is asserted in CI instead.
#:
#: An entry here that matches no file is an ERROR below -- a holdout that
#: protects nothing reads exactly like one with nothing to protect, right up
#: until the file it named is renamed and then stamped.
HOLDOUT: dict[str, str] = {}

#: 🔴 EXTENSIONS IS ITSELF A POPULATION CLAIM, AND NOTHING USED TO CHECK IT.
#: Measured at `fuel` d37e446e: its ratchet enumerates `*.rs` and reports
#: 834/834 forever while 203 tracked source files - 147 `.slang`, 20 `.glsl`,
#: 19 `.py`, 16 `.metal` - sit outside its population entirely, three of them
#: carrying Apple and third-party copyright with no SPDX at all.
#:
#: ⚠️ A PROPERTY ASSERTED OVER THE WRONG POPULATION IS STILL THE WRONG
#: POPULATION, and a scoped ratchet's number goes UP as the blind spot grows.
#:
#: This repo had the same shape at smaller scale: `.ts` x2 and `.sh` x1 sat
#: outside a `.py`-only list. So the list is now CHECKED: any tracked extension
#: that is source must be in EXTENSIONS or declined below, by name, with a
#: reason.
#:
#: ⚠️ IT DOES NOT FULLY CLOSE - `SOURCE_EXTENSIONS` is itself a hand-written
#: list, which is the defect class this whole gate keeps finding. What it does
#: is turn a SILENT omission into a LOUD one, which is the whole of the win.
SOURCE_EXTENSIONS = frozenset({
    ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".java",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".kt", ".scala", ".php",
    ".rb", ".pl", ".sh", ".bash", ".ps1", ".sql", ".lua", ".zig", ".dart",
    ".cu", ".cuh", ".cl", ".comp", ".vert", ".frag", ".geom", ".glsl", ".wgsl",
    ".hlsl", ".metal", ".slang",
})

#: Source extensions present in the tree and deliberately NOT stamped, each with
#: the reason. ⚠️ A DECISION ON THE RECORD, not an omission - and an entry here
#: naming an extension the tree does not have reds, like every other stale list
#: in this file.
NOT_STAMPED: dict[str, str] = {}

MARKER = "SPDX-License-Identifier:"


def normalise(identifier: str) -> str:
    """Canonical form for COMPARISON only. `Apache-2.0 OR MIT` and
    `MIT OR Apache-2.0` are the same grant, and one repo in this portfolio
    spells it each way. A false conflict standing next to a true one trains
    the reader to dismiss both."""
    text = identifier.strip()
    for joiner in (" OR ", " or "):
        if joiner in text:
            return " OR ".join(sorted(p.strip() for p in text.split(joiner)))
    return text


def declared(text: str) -> str | None:
    """The identifier a file declares, or None.

    Split on the marker, never a regex: an earlier `[\\w.-]+(?: OR [\\w.-]+)?`
    truncated `MIT OR Apache-2.0` to `MIT OR Apache`, which made every
    correctly-stamped file look wrong and would have had a second pass append a
    duplicate header to all of them.
    """
    for line in text.splitlines()[:HEADER_WINDOW]:
        if MARKER in line:
            value = line.split(MARKER, 1)[1].strip()
            for terminator in ("*/", "-->", "*)"):
                if value.endswith(terminator):
                    value = value[: -len(terminator)].strip()
            return value or None
    return None


def _git_z(root: pathlib.Path, *args: str, ok_codes=(0,)):
    """Run one git command with NUL-separated output. Returns (ok, names).

    ⚠️ ONE CALL SITE, BECAUSE COPIES OF AN INVOCATION ARE WHAT DRIFT. This file
    had three, and two of them disagreed with each other twice over:

      · `tools/spdx.py` used `-z` and this gate did not, so the SWEEPER and its
        own CHECKER read the same list two ways. A filename with a non-ASCII
        character came back quoted and octal-escaped, which is not a path that
        exists, and a perfectly good file reported as UNREADABLE.
      · `_git_z_all` accepted exit code 1 from `ls-files`, copied from the grep
        wrapper where 1 means "no matches". `ls-files` never returns 1.

    ⚠️ `ok_codes` IS EXPLICIT AT THE CALL SITE RATHER THAN HIDDEN HERE, because
    the two conventions are exactly what drifted. Measured, not assumed:

        git ls-files  with / without matches -> 0 ;  outside a repo -> 128
        git grep      no matches             -> 1 ;  with matches   -> 0

    So `ls-files` passes `ok_codes=(0,)` and `grep` passes `(0, 1)`, and a
    reader sees which convention a caller means without leaving the line.

    ⚠️ `ok=False` IS NOT AN EMPTY RESULT. Callers that treat an empty list as a
    PASS must be able to tell "ran, found nothing" from "could not run", and
    they could not when each site decided for itself.

    `git` is resolved with `shutil.which` rather than looked up through PATH at
    call time, so what runs depends on this file and not on the environment. The
    argv is fixed, `shell=False` is explicit, and the only interpolated value is
    this script's own parent directory.
    """
    git = shutil.which("git")
    if git is None:
        print("FAIL: no `git` on PATH.", file=sys.stderr)
        return False, []
    proc = subprocess.run(  # noqa: S603 - fixed argv, shell=False, see docstring
        [git, "-C", str(root), *args],
        capture_output=True, encoding=None, shell=False, check=False)
    if proc.returncode not in ok_codes:
        # ⚠️ stderr is REPORTED, never discarded. A failing git call and a git
        # call that found nothing both yield an empty list, and only one of them
        # should ever be reportable as a clean result.
        print(f"FAIL: git {args[0]}: "
              f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]}",
              file=sys.stderr)
        return False, []
    text = proc.stdout.decode("utf-8", "replace")
    return True, [n for n in text.split(chr(0)) if n]


def tracked_sources(root: pathlib.Path) -> list[str]:

    # ⚠️ RESOLVED ABSOLUTE, NOT "git". A bare name is looked up through PATH at

    # call time, so what runs depends on the environment rather than on this

    # file. The argv is fixed, there is no shell, and nothing here comes from a

    # caller - `root` is this script's own parent directory.

    ok, names = _git_z(root, "ls-files", "-z", "--",
                       *(f"*{e}" for e in EXTENSIONS), ok_codes=(0,))
    # ⚠️ An empty list from a FAILED call would read as "no source files",
    # which the MINIMUM_FILES floor would then blame on a broken glob.
    return names if ok else []




#: Files allowed to contain the word "copyright". ⚠️ A PATTERN, NOT A COUNT.
#: Licence texts contain it by definition; a changelog records licence changes;
#: this script discusses copyright in its own comments and so matches itself.
#: ⚠️ MATCHED ON THE BASENAME, NOT THE PATH. A substring test over the whole
#: path exempts anything beneath a directory whose name contains one of these -
#: `vendor/LICENSE-deps/foo.rs` would pass unexamined. Contrived here, where
#: nothing is vendored; not contrived in the three repos this gate also runs in,
#: two of which vendor third-party source.
COPYRIGHT_EXPECTED = ("LICENSE", "LICENCE", "COPYING", "CHANGELOG", "NOTICE",
                      "spdx_gate.py", "spdx.py", "preflight.py", "test_spdx.py")


def uncovered_extensions(root: pathlib.Path):
    """Tracked SOURCE extensions that EXTENSIONS does not cover, and stale
    NOT_STAMPED entries.

    ⚠️ THE RATCHET'S OWN POPULATION, CHECKED. Without this, adding `.wgsl`
    tomorrow escapes the gate silently and the pass rate goes UP - because the
    denominator never learns about the new files. That is how `fuel` reports
    834/834 while 203 tracked source files sit outside its scan.
    """
    ok, names = _git_z_all(root)
    if not ok:
        return None, None
    # ⚠️ `PurePosixPath.suffix`, NOT `rsplit(".")`. Filed as a LOW-RISK style
    # nitpick and it is a correctness bug - measured on real path shapes:
    #
    #     some.dir/file    rsplit -> ".dir/file"   suffix -> none
    #     a.b.c/README     rsplit -> ".c/readme"   suffix -> none
    #     .gitignore       rsplit -> ".gitignore"  suffix -> none
    #
    # A dot in a DIRECTORY name, or a dotfile with no extension, produced a
    # fabricated extension.
    #
    # ⚠️ `PurePosixPath` RATHER THAN `Path`, AND NOT FOR THE REASON IT LOOKS
    # LIKE. Measured on win32, where `Path` is `WindowsPath`: the two agree on
    # ALL 7 real path shapes tested, because WindowsPath accepts "/" happily.
    # They diverge on exactly one input - a filename containing a BACKSLASH
    # after a dot:
    #
    #     "a.rs" + chr(92) + "b"   WindowsPath -> ""   PurePosixPath -> ".rs\b"
    #
    # ⚠️ A BACKSLASH IN A GIT PATH IS PART OF THE FILENAME, NOT A SEPARATOR.
    # `git ls-files -z` emits POSIX paths raw on every platform, so the POSIX
    # reading is the correct one and WindowsPath would silently split a name.
    #
    # This is a fact about GIT, not about Python - and it is chosen for
    # correctness of interpretation, NOT because `Path` was observed to fail.
    # No repo here has such a filename, so the two are equivalent today.
    present = {pathlib.PurePosixPath(n).suffix.lower() for n in names}
    present.discard("")
    source_present = present & SOURCE_EXTENSIONS
    uncovered = sorted(source_present - set(EXTENSIONS) - set(NOT_STAMPED))
    # ⚠️ AGAINST EVERY PRESENT EXTENSION, NOT JUST THE SOURCE ONES.
    # NOT_STAMPED means "present in this tree and deliberately not
    # stamped"; the staleness question is whether it is STILL PRESENT,
    # not whether it is still classified as source. Comparing against
    # `source_present` reported vulkane's `.spv` and `.xml` as stale
    # while both sit in its tree - a decline of a NON-SOURCE extension
    # could never be recorded without redding.
    stale = sorted(set(NOT_STAMPED) - present)
    return uncovered, stale


def _git_z_all(root: pathlib.Path):
    """Every tracked path. Separate from `tracked_sources`, which is scoped."""
    # ⚠️ `ok_codes=(0,)` - `ls-files` NEVER exits 1. The `(0, 1)` this used to
    # carry was copied from the grep wrapper, where 1 means "no matches".
    return _git_z(root, "ls-files", "-z", ok_codes=(0,))


def _expected(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return any(tag in base for tag in COPYRIGHT_EXPECTED)


def survey_copyright(root: pathlib.Path) -> list[str]:
    """Tracked files carrying a copyright notice that are NOT expected to, or
    None if the survey COULD NOT RUN.

    ⚠️ `None` AND `[]` ARE DIFFERENT ANSWERS AND THE DIFFERENCE IS THE POINT.
    An empty list is the PASS condition here, so a survey that failed to run
    returning `[]` would be indistinguishable from a clean tree - and this
    gate's entire justification is that the holdout is CHECKED rather than
    asserted. A CHECK THAT SILENTLY NO-OPS ON FAILURE IS AN ASSERTION WITH A
    FUNCTION WRAPPED AROUND IT.

    ⚠️ THIS IS THE HOLDOUT'S JUSTIFICATION, MOVED OUT OF A COMMENT AND INTO CI.
    The comment used to state a COUNT: `-> 0` with a control of `-> 8`. That
    count drifted TWICE IN FOUR DAYS from two unrelated PRs, with neither author
    doing anything wrong - a changelog gained the word, then this script did.

    ⚠️ A COUNT CANNOT SURVIVE TREE GROWTH; A PROPERTY CAN. The thing the count
    was standing in for is "every -i copyright hit is a licence file or this
    script", and that is assertable. A COMMENT CANNOT GUARD - demonstrated twice
    inside this one file, first by the stale figure and then by the correction
    that vouched for it.

    ⚠️ AND IT FIRES EXACTLY WHERE THE HOLDOUT WOULD BE NEEDED. A new file
    carrying somebody else's copyright notice is precisely the case where a
    blanket sweep asserts a licence grant nobody made - `bs1770.rs` in `fuel`,
    Khronos's `vk.xml` in `vulkane`, Apple's kernels in `fuel`'s .metal files.
    """
    # ⚠️ `ok_codes=(0, 1)` HERE, and that is the difference the helper makes
    # visible: `git grep` exits 1 for "no matches", which is a RESULT and not
    # a failure. `ls-files` above passes (0,) because it never returns 1.
    ok, names = _git_z(root, "grep", "-l", "-i", "-z", "copyright",
                       ok_codes=(0, 1))
    if not ok:
        # ⚠️ None, not []. [] is the PASS condition for this check, so a
        # survey that COULD NOT RUN must not be reportable as a clean tree.
        return None
    return sorted(n for n in names if not _expected(n))


def audit(root: pathlib.Path, files: list[str]):
    """(missing, wrong, unreadable) over `files`, skipping HOLDOUT entries."""
    expected = normalise(LICENCE)
    missing, wrong, unreadable = [], [], []
    for rel in files:
        if rel in HOLDOUT:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append((rel, str(exc)))
            continue
        found = declared(text)
        if found is None:
            missing.append(rel)
        elif normalise(found) != expected:
            wrong.append((rel, found))
    return missing, wrong, unreadable


def report(files: list, missing: list, wrong: list, unreadable: list,
           stale: list) -> None:
    """Print the findings. Separated from deciding them so that changing how
    this reads cannot change what it concluded."""
    clean = len(files) - len(missing) - len(wrong) - len(unreadable) - len(HOLDOUT)
    print(f"{clean}/{len(files)} tracked source files declare {LICENCE!r}"
          + (f"  ({len(HOLDOUT)} held out)" if HOLDOUT else ""))
    rows = ([("MISSING", rel, "") for rel in missing]
            + [("DIFFERENT", rel, f" declares {found!r}") for rel, found in wrong]
            + [("UNREADABLE", rel, f": {why}") for rel, why in unreadable]
            + [("STALE HOLDOUT", rel,
                " matches no tracked file - it protects NOTHING") for rel in stale])
    for label, rel, suffix in rows:
        print(f"  {label}  {rel}{suffix}")


def explain(missing: list, wrong: list) -> None:
    """What to do about each kind of finding. Separate from the finding itself,
    because the remedies differ in KIND: one is mechanical, one is a decision."""
    if missing:
        print()
        print("Add the header as the FIRST line, above any `//!` inner docs:")
        print(f"    // {MARKER} {LICENCE}")
        print("A shebang stays on line 1 and the header goes below it.")
    if wrong:
        print()
        print("A file declaring a DIFFERENT licence is NOT a formatting error.")
        print("Changing it asserts a grant its author may not have made. Either")
        print("the declaration is right and the file belongs in HOLDOUT with a")
        print("reason, or it is wrong and that is a decision for the owner.")


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    root = pathlib.Path(__file__).resolve().parent.parent
    files = tracked_sources(root)

    if len(files) < MINIMUM_FILES:
        print(f"FAIL: found {len(files)} source files, expected at least "
              f"{MINIMUM_FILES}.")
        print("      This gate reports 100% compliance on an empty set, so it")
        print("      fails here instead. The GLOB is broken, not the tree.")
        return 1

    missing, wrong, unreadable = audit(root, files)
    stale = sorted(set(HOLDOUT) - set(files))
    # ⚠️ THE HOLDOUT'S JUSTIFICATION, CHECKED RATHER THAN ASSERTED IN PROSE.
    uncovered, stale_ns = uncovered_extensions(root)
    if uncovered is None:
        print("FAIL: could not enumerate the tree to check EXTENSIONS.",
              file=sys.stderr)
        return 1
    for ext in uncovered:
        print(f"  UNCOVERED EXTENSION  {ext} is tracked source and is not in "
              f"EXTENSIONS or NOT_STAMPED")
    for ext in stale_ns:
        print(f"  STALE NOT_STAMPED  {ext} is declined but no longer present")
    if uncovered:
        print()
        print("A source extension outside EXTENSIONS is invisible to this gate,")
        print("and the pass rate RISES as the blind spot grows. Add it to")
        print("EXTENSIONS, or to NOT_STAMPED with the reason it is excluded.")

    surveyed = survey_copyright(root)
    if surveyed is None:
        # ⚠️ The survey could not run. Refusing is the only honest outcome: a
        # clean report here would be a claim nobody measured.
        return 1
    unexpected = [n for n in surveyed if n not in HOLDOUT]
    report(files, missing, wrong, unreadable, stale)
    explain(missing, wrong)
    for rel in unexpected:
        print(f"  COPYRIGHT NOTICE  {rel} is not a licence file and is not "
              f"in HOLDOUT")
    if unexpected:
        print()
        print("A file carrying somebody else's copyright notice may not be")
        print("ours to license. Decide, then add it to HOLDOUT with the")
        print("reason, or to COPYRIGHT_EXPECTED if the match is incidental.")
    return 1 if (missing or wrong or stale or unreadable or unexpected
                 or uncovered or stale_ns) else 0


def self_test() -> int:
    """⚠️ The gate's own positive controls. A checker nobody has watched FAIL
    is a checker nobody has evidence works."""
    cases = [
        ("bare header", "// SPDX-License-Identifier: MIT OR Apache-2.0\n", "MIT OR Apache-2.0"),
        ("above inner docs", "// SPDX-License-Identifier: MIT OR Apache-2.0\n//! docs\n", "MIT OR Apache-2.0"),
        ("below a shebang", "#!/usr/bin/env python3\n# SPDX-License-Identifier: MIT OR Apache-2.0\n", "MIT OR Apache-2.0"),
        ("block comment", "/* SPDX-License-Identifier: MIT OR Apache-2.0 */\n", "MIT OR Apache-2.0"),
        ("nothing at all", "fn main() {}\n", None),
        # ⚠️ The truncation control. A regex-based reader returned `MIT OR
        # Apache` here and every correct file in the portfolio looked wrong.
        ("full dual identifier", "// SPDX-License-Identifier: MIT OR Apache-2.0\n", "MIT OR Apache-2.0"),
        # ⚠️ The self-counting control. This gate declares its own licence in
        # its own header AND names it in a string constant; a reader that
        # scanned the whole file would find the constant too.
        ("beyond the window", "\n" * 12 + "// SPDX-License-Identifier: MIT\n", None),
    ]
    failures = 0
    for name, text, expected in cases:
        got = declared(text)
        ok = got == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got!r}"
              + ("" if ok else f"  expected {expected!r}"))

    # ⚠️ CONTROLS FOR THE COPYRIGHT SURVEY'S CLASSIFIER. Needs no git: the part
    # that can silently rot is the PREDICATE, and `COPYRIGHT_EXPECTED` is a list
    # somebody will extend. A manual both-arms run proves the checker worked
    # that afternoon; nothing re-runs it when the list gains an entry.
    classifications = [
        ("LICENSE-MIT", True),
        ("crates/kiss-ref-core/LICENSE-APACHE", True),
        ("CHANGELOG.md", True),
        (".github/spdx_gate.py", True),
        ("src/lib.rs", False),
        ("crates/kiss-ops-vocab/src/lib.rs", False),
        # ⚠️ THE ONE THAT MATTERS, and it failed before basename anchoring: a
        # substring test over the whole PATH exempts everything beneath a
        # directory whose name contains a tag.
        ("vendor/LICENSE-deps/foo.rs", False),
        ("third_party/NOTICE-files/kernel.cu", False),
    ]
    for path, expected in classifications:
        got = _expected(path)
        ok = got == expected
        failures += not ok
        verb = "exempt" if expected else "examined"
        print(f"  {'ok  ' if ok else 'FAIL'}  {path} is {verb}")

    # ⚠️ CONTROLS FOR THE EXTENSION CENSUS. `uncovered_extensions` needs git,
    # but the part that ROTS is the path->extension derivation and the set
    # arithmetic, and neither does. A reviewer asked for this and was right:
    # a manual run proves it worked that afternoon; nothing re-runs it when
    # SOURCE_EXTENSIONS or NOT_STAMPED grows.
    #
    # The first three FAIL against the `rsplit(".")` form this replaced, so
    # they are controls rather than decoration.
    suffixes = [
        ("src/lib.rs", ".rs"),
        ("some.dir/file", ""),
        ("a.b.c/README", ""),
        (".gitignore", ""),
        ("x/y.tar.gz", ".gz"),
        ("crates/core/LICENSE-MIT", ""),
    ]
    for path, expected in suffixes:
        got = pathlib.PurePosixPath(path).suffix.lower()
        ok = got == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  suffix({path!r}) == {got!r}")

    # The census arithmetic, with the tree's extensions supplied directly.
    census = [
        ("a source ext outside the list is UNCOVERED",
         {".rs", ".md"}, (".py",), {}, [".rs"], []),
        ("a source ext IN the list is covered",
         {".rs", ".md"}, (".rs",), {}, [], []),
        ("a source ext DECLINED is covered",
         {".rs", ".md"}, (".py",), {".rs": "why"}, [], []),
        # ⚠️ THE CASE THAT WAS BROKEN: declining a PRESENT but NON-SOURCE
        # extension must not read as stale.
        ("a present NON-source decline is not stale",
         {".rs", ".spv"}, (".rs",), {".spv": "generated"}, [], []),
        ("an ABSENT decline IS stale",
         {".rs"}, (".rs",), {".slang": "gone"}, [], [".slang"]),
    ]
    for name, present, exts, not_stamped, want_unc, want_stale in census:
        source_present = present & SOURCE_EXTENSIONS
        unc = sorted(source_present - set(exts) - set(not_stamped))
        stl = sorted(set(not_stamped) - present)
        ok = unc == want_unc and stl == want_stale
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")

    equivalences = [("MIT OR Apache-2.0", "Apache-2.0 OR MIT", True),
                    ("Apache-2.0", "MIT OR Apache-2.0", False),
                    ("MIT", "MIT OR Apache-2.0", False)]
    for a, b, same in equivalences:
        ok = (normalise(a) == normalise(b)) == same
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {a!r} {'==' if same else '!='} {b!r}")

    # ⚠️ SUMMED, NOT WRITTEN DOWN. This line said "10 controls" while 18 ran,
    # for one commit - a stale count inside the run whose entire purpose is to
    # kill stale counts. A COUNT CANNOT SURVIVE ITS OWN LIST GROWING.
    total = (len(cases) + len(equivalences) + len(classifications)
             + len(suffixes) + len(census))
    print(f"{chr(10)}{'PASS' if not failures else 'FAIL'}: {total} controls, "
          f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
