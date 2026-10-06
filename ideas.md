# Baby Action AI — Design Direction

## Three possible approaches

### Theme Name: Soft Clinical Observatory
A warm, research-led interface with quiet cream surfaces, moss accents, and precise instrumentation details. It feels safe, considered, and trustworthy without looking like a hospital dashboard.

Probability: 0.07

### Theme Name: Playroom Signal
A bright, toy-inspired interface with primary color blocks, sticker-like cards, and expressive motion. It makes the technology approachable for caregivers and younger audiences.

Probability: 0.03

### Theme Name: Midnight Motion Lab
A dark, cinematic observatory with electric chartreuse signals and phosphor-like data trails. It treats pose recognition as a living sensor system and leans into the technical side of the product.

Probability: 0.08

## Chosen direction: Soft Clinical Observatory

### Design Movement
Contemporary editorial healthcare technology, interpreted through Swiss information design and warm natural materials rather than sterile blue enterprise UI.

### Core Principles
1. **Warm precision:** technical readouts sit inside calm, human-scaled surfaces.
2. **Instrumented asymmetry:** the layout favors offset panels, rails, and editorial margins over centered marketing blocks.
3. **Quiet evidence:** status, confidence, and limitations are always visible without shouting.
4. **Careful motion:** animations behave like measured sensor pulses—brief, responsive, and never playful for its own sake.

### Color Philosophy
The signature color is **signal moss** (`#B7D36B`), a soft high-visibility green that feels alive without resembling an alert. It sits against ink (`#17211C`), flax (`#F4F0E6`), and clay (`#C98A67`). The emotional intent is reassurance with technical credibility.

### Layout Paradigm
An editorial two-column composition: a narrow left rail anchors the brand and navigation, while the main field moves between a large hero statement, a live “observatory” panel, and offset evidence cards. Mobile collapses the rail into a top bar but preserves the instrument-panel rhythm.

### Signature Elements
1. A small “signal moss” pulse marker appears beside live statuses.
2. Thin calibration lines and coordinate labels frame key panels.
3. A hand-drawn-style landmark constellation motif repeats as a subtle background texture.

### Interaction Philosophy
Interactions should feel like operating a calm instrument. Hover states reveal one additional layer of context; toggles change the demo state immediately; buttons use short, tactile transitions. No interaction pretends the prototype is more reliable than it is.

### Animation
Use 180–240ms ease-out transitions for cards, buttons, and tabs. The demo signal uses a slow 2.4s opacity pulse, while landmark lines draw in once on first view. Respect reduced motion and keep all data transitions subtle.

### Typography System
Use **DM Serif Display** for large editorial headlines and **IBM Plex Sans** for interface copy, labels, and numeric readouts. Headlines should be short and sentence-cased; labels use compact uppercase with generous tracking. Numeric confidence values use IBM Plex Mono.

### Brand Essence
A privacy-first activity recognition lab for caregivers and builders who want clearer signals from everyday motion. Personality: **observant, grounded, humane**.

### Brand Voice
Headlines are direct and calm. CTAs are specific and low-pressure. Microcopy explains limitations plainly rather than using hype.

Example lines: “See the motion, not just the frame.” “A research prototype for noticing patterns earlier.”

### Wordmark & Logo
The mark is a rounded square aperture containing three connected landmark dots—head, shoulder, hip—arranged as a quiet diagonal. The wordmark is set in a custom-feeling wide sans with the “AI” treated as a small moss-colored instrument suffix.

### Signature Brand Color
**Signal moss — `#B7D36B`.**

## Style Decisions

- Keep the main experience light, warm, and editorial rather than dark neon or generic SaaS blue.
- Use generated visual assets only for the hero observatory image and a transparent landmark mark; use CSS and inline SVG for UI diagrams.
- Show the safety limitation in the interface, not buried in the footer.

## Style Decisions

- Error and empty states are treated as lost-signal observatory states using flax, ink, signal moss, calibration labels, and calm prototype-aware language.
- Primary actions use signal moss as the dominant accent across all pages; blue and red are avoided unless explicitly communicating external danger.
- Utility pages include at least one signature motif: the landmark aperture mark, moss pulse marker, coordinate label, or thin calibration framing.
