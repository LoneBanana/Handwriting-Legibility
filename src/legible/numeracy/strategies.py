"""Strategy hints: one short, concrete way to work out a fact the learner missed or was slow on.

Learners with dyscalculia often fall back on counting one by one, which is slow and
error-prone. Explicitly taught *derived-fact* strategies (use a fact you know to reach
one you don't) are what build fluency, so every hint names the strategy and walks it
through with the actual numbers, e.g. "9 + 6: move 1 from the 6 to make 10, then
10 + 5 = 15." Hints build on the skills that come earlier in the ladder (bonds to 10
before make-ten, doubles before near doubles, ×2/×5/×10 before the harder tables) and
never say what the learner did wrong.
"""

from __future__ import annotations

MINUS = "−"  # a real minus sign: clearer than a hyphen and read as "minus" by screen readers

# The order times tables are learned in (see ladder.py). A product is explained through
# whichever of its factors comes first here, so a hint only leans on tables already known.
TABLE_ORDER = (1, 2, 10, 5, 3, 4, 9, 6, 7, 8)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _listed(numbers: range) -> str:
    return ", ".join(str(n) for n in numbers)


def _halves(count: int) -> tuple[int, int]:
    """A split of ``count`` into two easy groups, the bigger first (5 -> 3 and 2)."""
    return (count + 1) // 2, count // 2


# ------------------------------------------------------------------ number sense


def subitize(count: int, layout: str) -> str:
    if count == 1:
        return "There is just 1 dot."
    if layout == "ten_frame":
        if count < 5:
            empty = _plural(5 - count, "empty box", "empty boxes")
            return f"A full row is 5. This row has {empty}, so it shows {count}."
        if count == 5:
            return "One full row of the ten-frame is 5."
        if count < 10:
            return f"A full row is 5, and {count - 5} more makes {count}."
        return "Two full rows make 10."
    big, small = _halves(count)
    if layout == "dice":
        if count <= 6:
            return f"This is the dice pattern for {count}. Try to see the whole pattern at once."
        return f"Two dice: {big} and {small} make {count}."
    if layout == "line":
        return f"Split the row into small groups: {big} and {small} make {count}."
    return f"Look for small groups, like {big} and {small}. Together they make {count}."


def compare(a: int, b: int) -> str:
    small, big = sorted((a, b))
    return f"When you count, {small} comes before {big}, so {big} is more."


def number_line(end: int, target: int, label_step: int) -> str:
    if target % label_step == 0:
        return f"Follow the arrow down to the line and read the number there: {target}."
    below = target - target % label_step
    above = below + label_step
    offset = (target - below) / label_step
    if offset <= 0.25:
        where = f"just after {below}"
    elif offset < 0.5:
        where = f"a little before halfway from {below} to {above}"
    elif offset == 0.5:
        where = f"halfway between {below} and {above}"
    elif offset < 0.75:
        where = f"a little past halfway from {below} to {above}"
    else:
        where = f"just before {above}"
    return f"The arrow is {where}, so it points to about {target}."


def bond(whole: int, part: int) -> str:
    missing = whole - part
    if whole == 10:
        if part == 0:
            return "An empty ten-frame needs all 10 dots, so 0 and 10 make 10."
        if part == 10:
            return "A full ten-frame has no empty boxes, so 10 and 0 make 10."
        return (
            f"A ten-frame with {_plural(part, 'dot')} has {_plural(missing, 'empty box', 'empty boxes')}, "
            f"so {part} and {missing} make 10."
        )
    if whole <= 10:
        if part == 0:
            return f"With nothing to start with, you need all {whole}: 0 and {whole} make {whole}."
        if part == whole:
            return f"{whole} is already all of it, so you need 0 more: {whole} and 0 make {whole}."
        verb = "is" if missing == 1 else "are"
        return (
            f"Hold up {whole} fingers and fold down {part}. {missing} {verb} still up, "
            f"so {part} and {missing} make {whole}."
        )
    return f"Count up from {part} to {whole}: that is {missing}, so {part} and {missing} make {whole}."


# ------------------------------------------------------------- adding, taking away


def add(a: int, b: int, *, prefer_make_ten: bool = False) -> str:
    """The most efficient strategy for ``a + b`` (or make-ten, when that skill is the focus)."""
    total, small, big = a + b, min(a, b), max(a, b)
    shown = f"{a} + {b}"
    if small == 0:
        return f"{shown}: adding 0 changes nothing, so {total}."
    if big == 10:
        if small == 10:
            return f"{shown}: 1 ten and 1 more ten make 2 tens, which is 20."
        return f"{shown}: 1 ten and {_plural(small, 'one')} make {total}."
    crosses_ten = total > 10 and big < 10
    if prefer_make_ten and crosses_ten:
        return _make_ten(shown, small, big, total)
    if a == b:
        if a <= 5:
            return f"{shown}: double {a} is {total}. Picture two rows of {a}."
        return f"{shown}: double 5 is 10 and double {a - 5} is {2 * (a - 5)}, so {total}."
    if total == 10:
        return f"{shown}: {a} and {b} make 10, like a full ten-frame."
    if big - small == 1 and small >= 2:
        return f"{shown}: double {small} is {2 * small}, and 1 more is {total}."
    if crosses_ten:
        return _make_ten(shown, small, big, total)
    return f"{shown}: start at {big} and count on {small}: {_listed(range(big + 1, total + 1))}."


def _make_ten(shown: str, small: int, big: int, total: int) -> str:
    need = 10 - big
    return f"{shown}: move {need} from the {small} to make 10, then 10 + {small - need} = {total}."


def subtract(a: int, b: int) -> str:
    difference = a - b
    shown = f"{a} {MINUS} {b}"
    if b == 0:
        return f"{shown}: taking away 0 leaves all {a}."
    if difference == 0:
        return f"{shown}: taking away all {a} leaves 0."
    if difference == 1:
        return f"{shown}: {b} and 1 more make {a}, so 1."
    if a == 10:
        return f"{shown}: {b} and {difference} make 10, so {difference} {'is' if difference == 1 else 'are'} left."
    if a < 10:
        if b <= 3:
            return f"{shown}: count back {b} from {a}: {_listed(range(a - 1, difference - 1, -1))}."
        return f"{shown}: count up from {b} to {a}: {_listed(range(b + 1, a + 1))}. That is {difference}."
    if b == 10:
        return f"{shown}: take away 1 ten and {difference} is left."
    if 10 < a < 20 and b < 10:
        ones = a - 10
        if b == ones:
            return f"{shown}: take away the {ones} in {a} and 10 is left."
        if b < ones:
            return f"{shown}: take {b} from the {ones} in {a} to leave {ones - b}, so {difference}."
        if difference <= 4:
            return (
                f"{shown}: count up from {b}. {10 - b} gets you to 10 and {ones} more gets you to {a}, "
                f"so {difference}."
            )
        return f"{shown}: take {ones} to get to 10, then take {b - ones} more to get to {difference}."
    return f"{shown}: count up from {b} to {a}. That is {difference}."


def skip_count(step: int, last: int) -> str:
    return f"Each number is {step} more than the one before: {last} + {step} = {last + step}."


# ------------------------------------------------------------- times and dividing


def multiply(a: int, b: int) -> str:
    """Explains ``a × b`` through the earlier-learned factor (``TABLE_ORDER``)."""
    shown = f"{a} × {b}"
    if a not in TABLE_ORDER or b not in TABLE_ORDER:
        return f"{shown}: {a} groups of {b} make {a * b}."
    factor = min((a, b), key=TABLE_ORDER.index)
    n = b if factor == a else a
    if factor == 1:
        return f"{shown}: times 1 keeps a number the same, so {n}."
    if factor == 2:
        return f"{shown}: double {n} is {n} + {n} = {2 * n}."
    if factor == 10:
        return f"{shown}: {n} tens make {10 * n}."
    if factor == 5:
        return f"{shown}: half of {n} × 10. Half of {10 * n} is {5 * n}."
    if factor == 3:
        return f"{shown}: double {n} is {2 * n}, plus one more {n} is {3 * n}."
    if factor == 4:
        return f"{shown}: double {n} is {2 * n}, and double again is {4 * n}."
    if factor == 9:
        return f"{shown}: {n} × 10 is {10 * n}, take away one {n} to get {9 * n}."
    if factor == 6:
        return f"{shown}: {n} × 5 is {5 * n}, plus one more {n} is {6 * n}."
    if factor == 7:
        return f"{shown}: {n} × 5 is {5 * n}, plus {n} × 2 is {2 * n}, so {7 * n}."
    return f"{shown}: {n} × 4 is {4 * n}, and double that is {8 * n}."


def divide(dividend: int, divisor: int) -> str:
    quotient = dividend // divisor
    return (
        f"{dividend} ÷ {divisor}: think {divisor} × ? = {dividend}. "
        f"{divisor} × {quotient} = {dividend}, so {quotient}."
    )
