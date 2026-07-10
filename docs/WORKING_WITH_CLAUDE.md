# Working with Claude: Finding Your Unknowns

A portable playbook for getting good results from a capable coding model (Claude
Fable 5 / Opus-class) on real projects. Distilled from Thariq Shihipar's
*"A Field Guide to Fable: Finding Your Unknowns"* (Anthropic, 2026) into
project-agnostic principles. Drop this file into any repo's `docs/` and adapt.

## The core idea: map vs. territory

> The map is not the territory.

- **The map** is everything you hand the model: the prompt, referenced files,
  specs, screenshots, skills, `CLAUDE.md`.
- **The territory** is the actual system: production constraints, past
  decisions, implicit taste, hidden business rules, edge cases, and the parts of
  the codebase nobody wrote down.

The model can only act on the map. Every gap between map and territory is an
**unknown** the model must silently guess through — and on a long task those
guesses compound into rework. Output quality is bottlenecked not by the model's
capability but by **your ability to close the map–territory gap**.

The people getting the most out of these models aren't writing *longer* prompts.
They run a **loop that surfaces assumptions** before, during, and after
implementation. Cheap discovery up front beats expensive fixes later.

## The four kinds of unknowns

| | You're aware of it | You're not aware of it |
|---|---|---|
| **You know it** | **Known knowns** — facts you put in the prompt | **Unknown knowns** — obvious context you'd recognize but never wrote down |
| **You don't know it** | **Known unknowns** — gaps you know are open | **Unknown unknowns** — things you haven't even considered |

The whole game is dragging the bottom-right cells up and to the left — turning
unknown unknowns into known unknowns into known knowns *before* they become bugs.

- **Known knowns** — already in the prompt. Nothing to do.
- **Known unknowns** — you know they're unresolved (which auth library? which
  timezone?). → **Interview** yourself/the model to close them.
- **Unknown knowns** — you'd instantly recognize the right answer but didn't
  think to state it (naming conventions, "we never touch that module", the way
  errors are surfaced). → **Brainstorm, prototype, and references** flush these out.
- **Unknown unknowns** — you haven't considered them at all (a concurrency edge
  case, a migration ordering constraint). → **Blindspot passes** surface these.

## The loop: techniques by stage

Each technique is a cheap probe that converts an unknown into something you can
act on. Prompts below are starting points — adapt the wording.

### Before implementation

**1. Blindspot Pass** — surface *unknown unknowns*.
Ask the model to attack your own request before it builds anything.
> "Before writing any code: what am I not thinking about here? List the edge
> cases, failure modes, assumptions, and design decisions this task depends on
> that I haven't specified. Flag anything ambiguous or risky."

**2. Brainstorm & Prototype** — flush out *unknown knowns* by seeing options.
You often don't know your own preference until you see the alternatives. Ask for
2–4 approaches, or a throwaway prototype, so your implicit taste becomes explicit.
> "Give me 3 distinct approaches to this, with trade-offs. Don't implement yet —
> I'll pick. Then throw together a rough prototype of the one I choose so we can
> see if it feels right."

**3. Interview** — close *known unknowns* by making the model ask *you*.
Flip the direction: instead of guessing, the model interrogates you — but only
about the things you alone hold: **intent, priorities, and how the real system
behaves.** It must never ask you to explain its own internals; if it needs to
know how the code works, it reads the code.
> "Interview me about this task. Ask only what you can't determine by reading the
> code yourself — my intent, my priorities, and how the real hardware/system
> actually behaves. One question at a time, most decision-critical first. Never
> ask me to explain the internals back to you. Don't start until the ambiguity is
> gone."

**4. References** — communicate what's hard to articulate.
Point at existing code, a similar feature, a screenshot, a competitor, a design.
"Make it like `X`" transfers a mountain of unstated requirements for free.
> "Follow the pattern in `path/to/similar_feature`. Match its structure, naming,
> error handling, and test style. Here's a screenshot of the behavior I want."

**5. Implementation Plan** — write the decisions down before coding.
Have the model produce a plan you can review and correct. Cheap to edit a plan;
expensive to edit a finished feature. This freezes the known knowns into a
contract you both work from.
> "Write an implementation plan: files you'll touch, the approach, key decisions
> and why, and what you're explicitly *not* doing. Wait for my sign-off before
> coding."

### During implementation

**6. Implementation Notes** — track deviations from the plan.
Reality diverges from the plan; make that divergence visible instead of silent.
> "As you work, keep a running log of any place you deviated from the plan, hit a
> surprise, or had to make a judgment call. Surface these instead of quietly
> deciding — I may want to weigh in."

### After implementation

**7. Explainer & Quiz** — validate that the map matched the territory.
Have the model explain what it built **in terms of observable behavior** — what
moves, what appears, the numbers you'd measure — not function names or internal
structure. Gaps in the explanation are gaps in the work.

The quiz runs in one direction only: the model quizzes you on **what you alone
can verify** — is this the behavior you intended, and does the real system
actually do it? It never quizzes you on the internals. A quiz you can't answer
from intent + the rig is a broken quiz; the fix is to restate it in observable
terms, not for you to go read the code.
> "Explain what you built in plain observable terms — what moves where, what I'd
> see on screen, the numbers I could measure — not function or class names. Tell
> me where you're least confident. Then quiz me only on what I can check from my
> intent and the real system: is this the behavior I wanted, and does the rig
> actually do it? Never quiz me on the internals."

## Operating principles

- **Discovery is cheap; rework is expensive.** Every blindspot pass, interview,
  prototype, and reference is a low-cost way to learn what you didn't know before
  it becomes a costly fix. Spend time up front.
- **Longer prompts aren't the fix — better-surfaced assumptions are.** The
  bottleneck is unstated context, not word count.
- **Make the model ask, not assume.** A model that interviews you or flags
  ambiguity is closing the gap; one that silently guesses is widening it.
- **Prefer relative/anchored instructions over absolute ones.** "Like this
  existing thing" carries more true intent than a paragraph of description.
- **Review the plan, not just the diff.** Catch a wrong assumption while it's one
  line in a plan, not a hundred lines across five files.
- **Surface deviations in real time.** Silent judgment calls during a long run
  are where compounding error hides.
- **The real system is ground truth.** Explanations and tests can be
  self-consistent and still wrong; verify the important parts against reality.

## On delegating without understanding every internal

If you've handed most of the implementation to the model, you may feel like a
fraud for not holding the internals in your head. You're not — and the fix isn't
to go learn every line. Two things resolve it:

- **You own the layer that can't be delegated.** Intent (what "correct" means)
  and ground truth (what the real system does) live only with you. That's the
  seat from which you catch the model being wrong — and you don't need to read
  its code to sit in it. *(The 300 nm → 100 nm oxide correction was caught this
  way: the map was checked against the wafer, not against the source.)*
- **Safety comes from legibility, not comprehension.** The real fragility isn't
  "I don't understand the internals" — it's "the *why* lives only in the model's
  head, where a fresh session, a collaborator, or future-you can't reach it."
  The antidote is **recoverability**: could someone reconstruct the reasoning
  when they need to? Every finding grounded on a number, image, or commit (see
  `FINDINGS.md`), every invariant pinned in `CLAUDE.md`, every commit that cites
  its reason, moves understanding out of the ephemeral and onto the page. That —
  not personal mastery of the code — is what makes the delegation robust.

You *can* hold a tractable middle layer without reading code: the **conceptual
spine** — the coordinate frames, the danger zones, the handful of invariants.
Hold the map's contour lines and you'll smell when something's off, even blind to
the individual rocks. That is all the interview and quiz will ever ask of you.

## Fast checklist

Before a non-trivial task, spend five minutes on:

- [ ] **Blindspot pass** — "what am I not thinking about?"
- [ ] **Interview** — let the model ask me the open questions
- [ ] **References** — point at the closest existing example
- [ ] **Plan** — get a written plan and correct it before any code
- [ ] **Notes** — ask for deviations to be surfaced, not swallowed
- [ ] **Explain & verify** — have it explain the result; check it on the real system

---

*Source: Thariq Shihipar, "A Field Guide to Fable: Finding Your Unknowns,"
Anthropic (2026).*
