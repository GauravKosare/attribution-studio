# Document 7: Frontend / Dashboard Tech Stack — Options Considered

This project ships with **Flask + a hand-written HTML/CSS/JS page + Plotly.js**
(`dashboards/server.py` + `dashboards/index.html`). That was a deliberate choice
for *this* project's goals, not the universally "best" stack for every attribution
dashboard. This document lays out the honest tradeoffs so the choice is legible
rather than assumed — pick differently if your goals differ.

## What this kind of project actually needs from a frontend
- Live recompute from user-chosen inputs (data source, journey rules, model
  parameters) — not a static report
- A handful of well-understood chart types: grouped bar, Sankey, a heatmap table,
  a simple bar-pair comparison — nothing needing custom rendering
- A form-heavy control surface (sliders, dropdowns, file upload, checkboxes)
- One page, not a multi-route application with auth/navigation/state persistence
- For a portfolio piece specifically: a result that *looks* like a real product,
  not a notebook — because that's what differentiates it in a stack of similar
  attribution-modeling projects

## The options, compared honestly

| Stack | Setup cost | Customization ceiling | Best fit | Real downside |
|---|---|---|---|---|
| **Streamlit** | Lowest — a Python script *is* the app | Low — you get Streamlit's component look, theming is limited, layout is constrained to its widget model | Fastest path to "it works" for personal/internal tools, or when the primary audience is other data people who don't care about visual polish | Every Streamlit app has a recognizable "Streamlit look"; hard to make it feel like a distinct product. Full reruns on every interaction can feel sluggish for a page with many controls |
| **Plotly Dash** | Low-medium — still Python-only, but callback model has a learning curve | Medium — more layout control than Streamlit, CSS-stylable, but you're still working inside Dash's component/callback paradigm | A data team building several related internal dashboards where Python-only is a hard requirement and Dash's callback graph fits the mental model | Verbose callback wiring for anything interactive; community/component ecosystem is smaller than React's; still reads as "a dashboard tool" rather than a bespoke app |
| **Flask/FastAPI + hand-written HTML/CSS/JS + Plotly.js** *(shipped here)* | Medium — no build step, but you write your own layout/interaction code | High — full control over every pixel and interaction; only bounded by your own CSS/JS | A single-page tool where you want it to look and feel deliberate, without taking on a JS build toolchain | You own more code: no component library means re-deriving common patterns (form state, loading states) by hand; scales poorly past one or two pages |
| **React/Vite (or Next.js) + a component/chart library (shadcn/ui, Recharts, Visx, Nivo) + FastAPI backend** | Highest — Node toolchain, build step, two codebases (frontend/backend) talking over an API | Highest — a real component model, reusable pieces, the best-in-class option if the project grows into multiple views/pages | A dashboard that's genuinely going to grow (more pages, auth, saved views, a team maintaining it) or where React/TypeScript is itself a skill you want to demonstrate | Meaningful setup and ongoing-maintenance overhead for a project this scoped; the extra structure doesn't pay for itself on a single-page tool |
| **Power BI / Tableau** | Low if you already know the tool; the modeling still happens in Python/SQL upstream | Medium — strong for enterprise BI conventions (drill-down, row-level security, scheduled refresh) but you're inside their UI paradigm | Roles where the job posting explicitly lists Power BI/Tableau — it's a different, non-code skill worth demonstrating on its own merits | Doesn't showcase frontend engineering at all; the interactive "any input" story this project tells (upload your own CSVs, live recompute) is awkward to build in either tool |
| **Observable Framework** | Low-medium — JS/data-notebook hybrid, git-based, static-site output | Medium-high — genuinely good for data-narrative pages, D3-native | A published, narrative-driven data report meant to be read more than operated | Not really built for a form-heavy "control panel" interaction model like this project's sidebar of sliders/toggles |

## Why this project shipped Flask + hand-written HTML/JS
Three reasons, in order of weight:
1. **The deliverable is one page**, not an app with multiple routes/views — the
   complexity a React/Next.js setup buys (routing, component reuse across pages,
   a design system) isn't needed here, so its cost isn't offset by anything.
2. **"Looks like a real product, not a notebook" was an explicit goal.** Streamlit
   and Dash are both excellent for iteration speed but visually legible as
   "a Streamlit app" — for a single showcase page, hand-written HTML/CSS gets a
   more distinctive result for comparable effort.
3. **No build toolchain** — no Node/npm/bundler in the loop, which matters on a
   Windows dev machine where keeping a JS toolchain healthy is its own tax, and
   the frontend is small enough that hand-writing it doesn't yet hurt.

## When you should choose differently
- **If this becomes a multi-page product** (separate views for exploration vs.
  the exec summary vs. an admin/data-upload panel), move to React/Next.js before
  the hand-written HTML file becomes unmaintainable — that threshold is usually
  "more than ~3 distinct pages" or "state needs to persist across views."
- **If the target audience is non-technical stakeholders who'll use it
  repeatedly**, invest in the React option for saved views, better mobile
  behavior, and a more forgiving component model for edge cases (empty states,
  error boundaries) than hand-written JS naturally gives you.
- **If the goal is fastest possible internal tool for yourself or a small data
  team**, Streamlit is the honest answer — this project's dashboard could be
  rebuilt in Streamlit in under an hour, at the cost of the visual distinctiveness
  described above. `dashboards/app.py` in this repo is exactly that version, kept
  for comparison.
- **If the job description explicitly names Power BI/Tableau**, build a `.pbix`/
  `.twbx` version too — it demonstrates a different, real skill that neither
  Streamlit nor a hand-written frontend touches.
