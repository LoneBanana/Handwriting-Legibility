"""Text to render as synthetic handwriting.

Real training should point ``corpus`` at a large plain-text file (one sentence or line
per row, e.g. extracted from Wikipedia or a school-writing corpus). The built-in
sentences below cover everyday writing (notes, lists, homework, messages) plus
pangrams, so smoke tests and small runs work out of the box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..vocab import normalize_text

BUILTIN_SENTENCES = """\
The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump!
Sphinx of black quartz, judge my vow.
Jackdaws love my big sphinx of quartz.
Remember to bring the permission slip on Monday.
My homework is due on Thursday after lunch.
Buy milk, eggs, bread, apples and two onions.
Call grandma before dinner and ask about Sunday.
The meeting moved to room 204 at half past three.
I think the answer to question four is seventeen.
Please feed the cat and water the plants.
We read two chapters of the book in class today.
The science project needs a poster and a short talk.
Dear Ms. Patel, thank you for helping me with maths.
Practice the piano for twenty minutes every day.
Take the bus at 8:15 and get off at Park Street.
My favourite animal is the octopus because it is clever.
The volcano erupted and lava flowed down the mountain.
Photosynthesis turns sunlight, water and air into sugar.
Write a paragraph about your summer holiday.
Don't forget your coat, it might rain later.
Soccer practice is cancelled because the field is wet.
The library closes early on Fridays at five.
I would like to learn how to play the guitar.
Mix the flour and sugar, then add the butter slowly.
Our class went to the museum to see the dinosaurs.
Tomorrow we have a spelling test on long vowel words.
The train was late so I missed the start of the film.
Can you help me carry these boxes upstairs?
The answer is 3 x 4 = 12, so the total is twelve.
Next week we start a new topic about the solar system.
I left my glasses on the kitchen table again.
Doctor's appointment on the 14th at 10:30 am.
Thank you for the birthday card, it made me smile.
The river flows from the hills down to the sea.
My brother is seven years old and loves trains.
We need to finish the group report by Wednesday.
History notes: the castle was built in 1086.
Remember: capital letters start every sentence.
The recipe needs 250 g of flour and 2 eggs.
I felt nervous before the race, but I did my best.
Clean your room before you go out to play.
The weather today is sunny with a light breeze.
Ask the teacher if we can use calculators.
Gym kit, water bottle, lunch box, reading book.
Our team won the quiz by just one point!
Plants need light, water, warmth and good soil.
I am going to visit my aunt during the break.
Please sign and return the form by Friday.
The dog chased the ball across the wet grass.
In maths we learned how to add fractions.
My password hint is my first pet's name.
Fix the bike tyre and oil the chain.
Rain is forecast for the whole weekend.
She painted the fence blue and the door yellow.
Water boils at 100 degrees Celsius at sea level.
Our address is 42 Maple Road, Flat 3.
The concert starts at seven, so leave by six.
I can't wait for the school trip next month.
Read pages 31 to 45 and answer the questions.
The cake needs forty minutes in the oven.
He scored three goals in the second half.
Bring a packed lunch and a waterproof jacket.
We saw a rainbow after the storm passed.
The shop sells fresh fruit, vegetables and flowers.
Every Tuesday I help my dad cook dinner.
The moon goes around the Earth about once a month.
Turn left at the lights, then take the second right.
Is it okay if I hand in my essay on Monday?
The orchestra played loudly and the crowd cheered.
My goal this term is to write more neatly.
Electricity flows through the wire to the bulb.
Glue, scissors, coloured pencils and a ruler.
The puppy fell asleep on the warm blanket.
We planted tomatoes, beans and sunflowers.
A triangle has three sides and three angles.
Text me when you get home safely.
The bridge was closed for repairs all week.
I need new shoes before the start of term.
Ancient Egyptians built pyramids as tombs.
Recycling day is every other Thursday.
The film was funny, but the ending was sad.
Hello! How are you feeling today?
Notes: test on chapters 5-7, bring a pencil.
The fire alarm test is at 11 o'clock.
Measure the length, width and height of the box.
The bakery on the corner makes great bagels.
Who wants to come swimming on Saturday?
"""

_WORDS = (
    "apple table window garden pencil yellow orange purple river mountain "
    "morning evening friend family teacher school number letter answer question "
    "quickly slowly happy careful bright quiet heavy simple strange wonderful"
).split()


class TextCorpus:
    """Samples line-length snippets of text, plus occasional numbers, dates and times."""

    def __init__(self, lines: list[str], p_structured: float = 0.08):
        cleaned = [normalize_text(line) for line in lines]
        self.lines = [line for line in cleaned if line]
        if not self.lines:
            raise ValueError("text corpus is empty")
        self.p_structured = p_structured

    @classmethod
    def builtin(cls) -> "TextCorpus":
        return cls(BUILTIN_SENTENCES.splitlines())

    @classmethod
    def from_file(cls, path: str | Path, max_lines: int = 5_000_000) -> "TextCorpus":
        lines = []
        with Path(path).open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                lines.append(line)
                if len(lines) >= max_lines:
                    break
        return cls(lines)

    @classmethod
    def load(cls, path: str | Path | None) -> "TextCorpus":
        return cls.from_file(path) if path else cls.builtin()

    def sample(self, rng: np.random.Generator, min_chars: int = 3, max_chars: int = 60) -> str:
        if rng.random() < self.p_structured:
            return _structured(rng)
        line = self.lines[int(rng.integers(len(self.lines)))]
        if len(line) <= max_chars:
            return line
        words = line.split(" ")
        target = int(rng.integers(max(min_chars, 8), max_chars + 1))
        start = int(rng.integers(len(words)))
        out: list[str] = []
        length = 0
        for word in words[start:] + words[:start]:
            extra = len(word) + (1 if out else 0)
            if out and length + extra > target:
                break
            out.append(word)
            length += extra
        text = " ".join(out)[:max_chars].strip()
        return text if len(text) >= min_chars else line[:max_chars].strip()


def _structured(rng: np.random.Generator) -> str:
    kind = int(rng.integers(6))
    if kind == 0:
        return f"{int(rng.integers(1, 29))}/{int(rng.integers(1, 13))}/{int(rng.integers(1990, 2040))}"
    if kind == 1:
        return f"{int(rng.integers(1, 13))}:{int(rng.integers(0, 60)):02d} {'am' if rng.random() < 0.5 else 'pm'}"
    if kind == 2:
        return f"£{int(rng.integers(1, 200))}.{int(rng.integers(0, 100)):02d}" if rng.random() < 0.3 else \
            f"${int(rng.integers(1, 500))}.{int(rng.integers(0, 100)):02d}"
    if kind == 3:
        a, b = sorted((int(rng.integers(2, 13)), int(rng.integers(2, 13))), reverse=True)
        op = str(rng.choice(["+", "-", "x"]))
        result = {"+": a + b, "-": a - b, "x": a * b}[op]
        return f"{a} {op} {b} = {result}"
    if kind == 4:
        return f"{int(rng.integers(1, 300))} {str(rng.choice(_WORDS)).capitalize()} Street"
    words = rng.choice(_WORDS, size=int(rng.integers(2, 6)))
    return " ".join(str(w) for w in words)
