# CLI Example: Sci-Fi Library

A walkthrough showing how to use the `x64rag` CLI to analyze, classify, check consistency, and evaluate a series of science fiction novels — helping you research and write in a shared universe.

The `scifi-library/` folder contains 8 books from the "Sands of Araxis" saga. Each is a markdown file with several chapters (~800-1200 words each).

## 1. Install and Configure

```bash
uv add "x64rag[cli]"
x64rag reasoning init
```

Edit `~/.config/x64rag/.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Edit `~/.config/x64rag/config.toml`:

```toml
[language_model]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
```

Suppress verbose BAML logging (optional):

```bash
export BAML_LOG=error
```

Verify:

```bash
x64rag reasoning status
```

```
Config: /home/you/.config/x64rag/config.toml
.env: /home/you/.config/x64rag/.env
Provider: anthropic
Model: claude-sonnet-4-5-20250929
LLM: connected

Ready.
```

## 2. Analyze a Book

Analyze Book 1 to understand its structure, themes, and narrative qualities:

```bash
x64rag reasoning analyze --file scifi-library/book-01-sands-of-araxis.md --summarize \
  --dimensions dimensions.json
```

```
Intent: To introduce a science fiction narrative establishing a protagonist's dangerous
  political assignment to control a desert planet that is the sole source of a critical
  resource, while building a complex world with ecological, political, and cultural
  systems. (95% confidence)

Summary: Seventeen-year-old Kael Vandris learns his family has been granted stewardship
  of Araxis, a lethal desert planet that is the only source of chronodust — a substance
  essential for interstellar navigation. The assignment appears to be a political death
  sentence, as the previous ruling house was completely destroyed, and survival requires
  mastering the harsh environment and its native inhabitants' water-conservation techniques.

Dimensions:
  tension: 0.75 (90%)
    High political stakes established immediately (previous rulers extinguished, potential
    trap by Emperor), combined with environmental dangers (extreme heat, deadly creatures,
    resource scarcity).
  pacing: moderate (85%)
    The narrative balances plot advancement (inheritance announcement, journey to Araxis,
    beginning training) with substantial worldbuilding exposition.
  worldbuilding: 0.9 (95%)
    Extremely dense with specific details: political structure (Hegemony, Emperor, House
    system), unique resource economics (chronodust for navigation, water as currency),
    environmental specifications (70°C, 3% humidity, 4000km desert).
  character_depth: 0.5 (75%)
    Moderate depth shown primarily through dialogue and reactions. Kael demonstrates
    analytical thinking, his mother shows political sophistication and emotional control.
    However, internal emotional states are minimally explored.
```

The `dimensions.json` file defines what to score:

```json
[
  {"name": "tension", "description": "Level of narrative tension and conflict", "scale": "0.0-1.0"},
  {"name": "pacing", "description": "Speed of plot progression", "scale": "slow/moderate/fast"},
  {"name": "worldbuilding", "description": "Density of world details introduced", "scale": "0.0-1.0"},
  {"name": "character_depth", "description": "Emotional and psychological depth of characters", "scale": "0.0-1.0"}
]
```

## 3. Classify Books by Theme

Create categories for the types of content in your series:

```json
[
  {"name": "action", "description": "Combat, chase, or physical conflict scenes"},
  {"name": "political_intrigue", "description": "Power plays, alliances, betrayals, and court machinations"},
  {"name": "exploration", "description": "Discovery of new places, technologies, or knowledge"},
  {"name": "character_drama", "description": "Interpersonal conflict, relationships, and emotional arcs"},
  {"name": "worldbuilding_exposition", "description": "Establishing setting, history, culture, or technology"},
  {"name": "crisis_and_survival", "description": "Characters facing existential threats or impossible choices"}
]
```

Classify each book:

```bash
x64rag reasoning classify --file scifi-library/book-01-sands-of-araxis.md --categories categories.json
```

```
Category: worldbuilding_exposition (95%)
Strategy: llm
Reasoning: The text primarily establishes the setting, culture, and key elements of the
  world including the planet Araxis, its extreme environment, the native Araxeen people,
  and the political/economic importance of chronodust. While there is a political element
  in the inheritance announcement, the bulk of the content focuses on describing the
  world's physical characteristics, history, and inhabitants.
Runner-up: political_intrigue (40%)
```

```bash
x64rag reasoning classify --file scifi-library/book-05-the-hegemony-falls.md --categories categories.json
```

```
Category: political_intrigue (85%)
Strategy: llm
Reasoning: The text centers on the Emperor's decision regarding fold travel and the
  resulting political split of the Hegemony into factions, with advisors debating the
  validity of information and different systems forming alliances based on their
  interests. While there are elements of crisis and worldbuilding, the primary focus is
  on power dynamics, alliances, and the political consequences of the Emperor's choice.
Runner-up: crisis_and_survival (65%)
```

You can batch-classify all books to see the arc of your series:

```bash
for book in scifi-library/book-*.md; do
  name=$(basename "$book" .md)
  category=$(x64rag reasoning --json classify --file "$book" --categories categories.json | jq -r '.category')
  echo "$name: $category"
done
```

```
book-01-sands-of-araxis: worldbuilding_exposition
book-02-the-chronodust-war: action
book-03-keepers-of-the-fold: exploration
book-04-the-sandweaver-codex: exploration
book-05-the-hegemony-falls: political_intrigue
book-06-exile-of-house-vandris: character_drama
book-07-the-substrate-war: crisis_and_survival
book-08-the-new-navigators: worldbuilding_exposition
```

This tells you: Books 1 and 8 bookend the series with worldbuilding, the middle is action/exploration/politics, and Book 6 is the emotional core. If you're planning Book 9, you can see what's missing.

## 4. Check Consistency

The compliance command checks whether new writing contradicts established canon. Use an earlier book as the reference document:

```bash
x64rag reasoning compliance --file scifi-library/book-08-the-new-navigators.md \
  --references scifi-library/book-01-sands-of-araxis.md
```

```
Compliance: FAIL (0.30)

Violations (8):
  [high] narrative_continuity — The text begins 'Five years after the moratorium' but
    no moratorium is mentioned or established in the reference document.
    Suggestion: Provide context that bridges from the reference material, or ensure the
    reference document includes this event.
  [high] technological_consistency — The text introduces 'Resonance discipline',
    'Keepers', and 'symbiotic fold' technology that has no basis in the reference
    material. The reference establishes chronodust navigation but provides no foundation
    for these alternative methods.
    Suggestion: Show how these technologies evolved from the chronodust navigation system
    described in the reference.
  [high] world_building_consistency — The 'Resonance Academy' and the 'Keepers'
    organization are introduced without any precedent in the reference material.
    Suggestion: Add foundation for the Keepers in earlier books, or explain their origin
    and relationship to established institutions like the Cognitive Order.
  [high] world_building_consistency — The reference establishes sandweavers as dangerous
    creatures. The evaluated text presents them as cooperative beings that can be
    'acclimated' to human presence, which contradicts the established characterization.
    Suggestion: Add foundation in earlier books for sandweaver intelligence and potential
    cooperation.
  [medium] narrative_continuity — The text references 'the Consortium way' without any
    mention of a Consortium in the reference material.
  [medium] technological_consistency — The reference describes chronodust as making
    'interstellar navigation possible' but doesn't explain the mechanism. The evaluated
    text introduces 'fold navigation' and 'substrate topology' as established concepts.
  [medium] narrative_continuity — The text mentions 'the Cognitive Order's three pillars'
    (Pattern, Control, and Projection) which are not established in the reference material.
  [low] character_consistency — Kael is 17 in the reference and has just arrived on
    Araxis. The evaluated text shows him as an established leader, which is plausible but
    represents a significant time jump.

Reasoning: The text introduces significant narrative elements, technologies, and plot
  developments that have no foundation in the reference material. The text jumps forward
  five years and references events, institutions, and concepts that are completely absent
  from the reference chapters.
```

This is exactly the kind of feedback a writer needs when Book 8 contradicts Book 1. You can now go back and plant the seeds for "Keepers", "the moratorium", and sandweaver cooperation in the earlier books.

But checking against Book 1 alone is unfair — Book 8 builds on 7 books of established canon. Use `--references` to check against multiple books at once:

```bash
x64rag reasoning compliance --file scifi-library/book-08-the-new-navigators.md \
  --references scifi-library/book-01-sands-of-araxis.md \
  --references scifi-library/book-07-the-substrate-war.md
```

```
Compliance: FAIL (0.85)

Violations (4):
  [medium] technological_consistency — The reference states chronodust is 'generated by the
    passage' of sandweavers through dunes, implying a byproduct of their movement. The
    evaluated text describes 'chronodust crystal' chambers, which seems inconsistent with
    the dust/powder nature implied in the reference.
    Suggestion: Clarify that the chamber is carved from compressed or crystallized chronodust
    deposits, or use terminology like 'chronodust-infused stone'.
  [medium] world_building_consistency — The text mentions 'Keepers' as a distinct tradition
    with their own 'Resonance discipline' spanning generations, but this group is not
    introduced in the reference material.
    Suggestion: Either introduce the Keepers in earlier reference material, or provide
    context explaining who they are.
  [low] terminology_consistency — The text introduces 'three pillars' of the Cognitive Order
    (Pattern, Control, and Projection) without these being established in the reference.
    Suggestion: Either remove the specific pillar names or add a brief explanation that
    these pillars were formalized after the events of the earlier books.
  [low] narrative_logic — The text states the Academy opened 'five years after the
    moratorium' but doesn't clarify how the transition period unfolded.
    Suggestion: Add a brief reference to the moratorium's completion to provide clearer
    timeline continuity.

Dimension Scores:
  world_building_consistency: 0.90
  character_continuity: 0.95
  terminology_consistency: 0.75
  narrative_logic: 0.85
  thematic_alignment: 0.90
```

Score jumped from 0.30 (Book 1 only) to 0.85 (Books 1 + 7) — with Book 7 as context, concepts like "the moratorium", "Consortium", and "fold navigation" are now established canon, not violations. The remaining issues are minor terminology gaps.

## 5. Analyze Chapter Progression

Use `analyze-context` to track how narrative dimensions evolve across chapters within a book. Each chapter is a segment in a JSON array:

```bash
x64rag reasoning analyze-context --file chapters-book-08.json --summarize --dimensions dimensions.json
```

```
Intent: Narrative conclusion depicting the successful transformation of a society from
  exploitative resource extraction to symbiotic cooperation between species (95% confidence)

Summary: This epilogue chronicles the establishment of the Resonance Academy on Araxis,
  where humans learn to cooperatively navigate space with sandweavers rather than exploiting
  them, culminating in a peaceful federation that replaces extraction with partnership.

Dimensions:
  tension: 0.2 (90%)
    Minimal conflict present; the text focuses on resolution and peaceful outcomes. Early
    mentions of integration challenges provide slight tension, but overall tone is
    harmonious and reflective.
  pacing: slow (85%)
    The narrative spans decades (40+ years) with contemplative, reflective passages. Time
    moves in large jumps with detailed descriptions of processes and philosophical insights.
  worldbuilding: 0.85 (90%)
    Dense with specific details about the Resonance Academy curriculum, symbiotic fold
    navigation mechanics, political structures (Hegemony to Accord), and the twelve-system
    federation.
  character_depth: 0.7 (80%)
    Characters are presented through their life choices, final words, and philosophical
    reflections. Kael's self-assessment as 'adequate translator,' Senna's trust despite
    lack of understanding, and Dhamira's patient longevity reveal meaningful depth.

Intent Shifts:
  [1] Establishing educational framework → Demonstrating practical application
    Chapter 1 focuses on the Academy's founding, while Chapter 2 shifts to showing the
    first successful symbiotic fold navigation as proof of concept.
  [2] Demonstrating practical application → Providing long-term resolution and legacy
    Chapter 2 demonstrates immediate success, while Chapter 3 jumps decades forward to
    show lasting societal transformation and philosophical summation.

Escalation: no
Resolution: resolved
```

The `chapters-book-08.json` represents each chapter as a `{role, text}` segment:

```json
[
  {"role": "chapter_1", "text": "Five years after the moratorium..."},
  {"role": "chapter_2", "text": "The first symbiotic fold..."},
  {"role": "chapter_3", "text": "Kael Vandris governed Araxis for forty more years..."}
]
```

This tells the writer: tension drops from moderate to low across the final book, pacing slows dramatically, but worldbuilding density stays high. If Book 8 needs more tension, the data shows exactly where to add it.

## 6. Evaluate Revisions

After rewriting a chapter, compare the revision against the original:

```bash
x64rag reasoning evaluate \
  --generated scifi-library/book-08-the-new-navigators.md \
  --reference scifi-library/book-01-sands-of-araxis.md \
  --strategy judge
```

```
Score: 0.35 (low)
Judge: 0.35
  The generated output appears to be from a later section of the same story, showing
  narrative progression and thematic consistency with the reference material. However,
  it suffers from significant context issues — it references events, characters, and
  concepts that are not established in the provided text. The writing quality is competent
  with clear prose and coherent world-building, but it reads like a sequel chapter without
  the necessary foundation.
```

This is more useful when comparing two versions of the same chapter — the original draft vs a revision — to see if the rewrite actually improved.

## 7. Piped Input for AI Assistants

The CLI is designed for AI models working alongside you. When piped, output is automatically JSON:

```bash
echo "The sandweavers emerged from the dunes at sunset, their crystalline bodies \
refracting the last light into prismatic arcs across the desert floor. Kael watched \
from the observation platform, knowing that these creatures held the key to everything." \
  | x64rag reasoning analyze --summarize
```

```json
{
  "primary_intent": "Narrative storytelling — establishing a scene with world-building elements and tension",
  "confidence": 0.95,
  "summary": "Kael observes mysterious crystalline creatures called sandweavers at sunset, harboring dangerous knowledge about chronodust and the fold that could destabilize peace if discovered.",
  "dimensions": {},
  "entities": [],
  "retrieval_hints": []
}
```

An AI assistant can parse this and act on it — checking every paragraph you write for consistency, classifying scenes as you draft them, or analyzing pacing across chapters.

## 8. A Writing Research Session

Here's a realistic workflow — you're writing the final chapter of Book 8 and need help:

```bash
# What's the narrative arc of Book 7 going into the finale?
x64rag reasoning analyze --file scifi-library/book-07-the-substrate-war.md --summarize \
  --dimensions dimensions.json

# What category should the final chapter be? Check what's missing.
for book in scifi-library/book-*.md; do
  echo "$(basename $book): $(x64rag reasoning --json classify --file $book --categories categories.json | jq -r .category)"
done

# Write your chapter, then check it against the full series canon
x64rag reasoning compliance --file final-chapter-draft.md \
  --references scifi-library/book-01-sands-of-araxis.md \
  --references scifi-library/book-07-the-substrate-war.md \
  --references scifi-library/book-08-the-new-navigators.md

# Revise and evaluate improvement
x64rag reasoning evaluate --generated final-chapter-v2.md --reference final-chapter-draft.md \
  --strategy judge
```

The CLI is the simple, global interface — one command per operation, no code required. For batching, custom pipelines, or embedding in applications, use the Python SDK directly.
