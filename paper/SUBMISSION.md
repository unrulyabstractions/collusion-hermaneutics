# Submission checklist: Apart Research, Swarm Alignment 2026

Source: the sprint page https://apartresearch.com/sprints/swarm-alignment-2026, read 2026-11-08 before 19:24 UTC. Submit through the "Submit your project" form on that page (a modal on the page itself). Deadline, in the page's words: "Closes Nov 8, 11:59 PM AoE (Anywhere on Earth (UTC−12))", shown as Nov 9, 11:59 AM UTC. Length rule, in the page's words: "Maximum 8 pages, not counting references and appendices."

| form field | value | status |
|---|---|---|
| Project title * | Hermeneutics for swarm collusion monitoring | set (rename if you wish) |
| Project summary * | the text of `paper/build/abstract.txt` (222 words; 231 with hyphenated words split) | ready |
| PDF report * (up to 25 MB) | `paper/build/main.pdf` (main text pages 1 to 8, references from page 8, then the appendices; `paper/src/build.sh` prints the page map after each build), with `paper/build/supplement.pdf` beside it | built; your edit pass pending |
| Are you interested in publishing this project? * | Yes / No | your call |
| Tracks (choose every track that fits) | Track 4: Multi-Agent Oversight and Control; Track 2: The Science of Agent Networks | ready |
| Presentation recording (optional) | none | skip or record |
| Project code (optional) | https://github.com/unrulyabstractions/collusion-hermaneutics | ready (public) |
| Slides (optional) | none | skip |
| Project image (optional) | none | skip, or ask for a figure |
| Additional material (optional) | none | skip |
| Team name * | your choice | you |
| Location * | your choice | you |
| Team member 1: Name *, Email *, Discord username, Google Scholar profile | Ian Rios-Sialer; your email; optional fields | you |

Before you press submit:

1. Finish your edit pass on `paper/src/main_template.md` and `paper/src/supplement_template.md` (the files in `paper/build/` are generated), then run `.venv/bin/python paper/src/gen_report.py && paper/src/build.sh` (or ask me) and read the page line it prints: the main text must stay within 8 pages.
2. Read the LLM Usage Statement at the end of the PDF; it says the author reviewed the text.
3. Check the title and the affiliation line on page 1.
4. `git push` so the public repository matches the PDF.
5. The form says to check that the PDF is the final version before you submit.
