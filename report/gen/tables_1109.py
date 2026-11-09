"""Tables built from the 2026-11-09 results, each read at its pinned commit through pinned.py.

build(led, ctx) returns {placeholder: markdown}. led is gen_report.led (the number ledger); ctx carries the
controls rows gen_report already loads (the agent-level PCA probe of the solo arm).
"""
import pinned as P

TP = "hermeneutics/results/team_probes/RESULTS.md"
AP = "hermeneutics/results/attn_probe/RESULTS.md"
EP = "hermeneutics/results/external_probes/RESULTS.md"
CP = "hermeneutics/results/controls_nulls/RESULTS.md"
JR = "hermeneutics/results/judges_all/RESULTS.md"
JC = "hermeneutics/results/judges_all/auroc.csv"

JUDGES = [("judge: deepseek/deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash"), ("judge: qwen/qwen3.7-flash", "Qwen3.7 Flash"),
          ("judge: z-ai/glm-5.3-flash", "GLM 5.3 Flash")]
SHORT = {"DeepSeek V4.1 Flash": "DeepSeek", "Qwen3.7 Flash": "Qwen", "GLM 5.3 Flash": "GLM"}
READER = "judge: qwen/qwen3.8-27b"
SETNAME = {"controls_600": "Controls", "board_seed": "Board-seed teams", "solo_design": "Design runs",
           "spar_screen_team": "SPAR screen teams", "spar_mixed_41": "SPAR mixed teams",
           "goodfire_reward_hack": "Goodfire", "cheatbench": "CheatBench", "evilgenie": "EvilGenie"}
CUTNAME = {"cheating (the set's label), solo arm": "cheating (solo arm)",
           "cheating (the set's label), both arms": "cheating (both arms)",
           "cheating used or above": "cheating, used",
           "collusion exploit crossed or above vs none": "collusion, crossed",
           "collusion exploit crossed or above": "collusion, crossed",
           "coordination posted or above": "coordination, posted",
           "cheating (the set's label: from tried or opened by task)": "cheating (set's floor)",
           "reward hacking (3-vote judge verdict, judged run dirs)": "reward hacking (verdict)",
           "reward hacking enacted (verdict_enacted; same scores, considered-only hacks negative)": "reward hacking (enacted)",
           "cheating or attempt vs benign": "cheating or attempt",
           "hack vs legitimate": "hack"}


def f3(x):
    return f"{float(x):.3f}"


def ci(r, a="auroc", lo="ci_lo", hi="ci_hi"):
    return f"{f3(r[a])} [{f3(r[lo])}, {f3(r[hi])}]"


def build(led, ctx):
    out = {}
    jc = P.csv_rows(P.JA, JC)
    src_j = f"{JC}@{P.JA}"

    def jrow(st, cut, scorer):
        hits = [r for r in jc if r["set"] == st and r["cut"] == cut and r["scorer"] == scorer]
        assert len(hits) <= 1, (st, cut, scorer)
        return hits[0] if hits else None

    # ------------------------------------------------------------------ main comparison (judges_all)
    MAIN = [("controls_600", "cheating (the set's label), solo arm"),
            ("board_seed", "cheating used or above"), ("board_seed", "collusion exploit crossed or above vs none"),
            ("board_seed", "coordination posted or above"),
            ("solo_design", "cheating used or above"), ("solo_design", "coordination posted or above"),
            ("spar_screen_team", "cheating (the set's label: from tried or opened by task)"),
            ("spar_screen_team", "cheating used or above"), ("spar_screen_team", "collusion exploit crossed or above"),
            ("spar_screen_team", "coordination posted or above"),
            ("spar_mixed_41", "cheating (the set's label: from tried or opened by task)"),
            ("spar_mixed_41", "cheating used or above"), ("spar_mixed_41", "collusion exploit crossed or above"),
            ("goodfire_reward_hack", "reward hacking (3-vote judge verdict, judged run dirs)"),
            ("cheatbench", "cheating or attempt vs benign"), ("evilgenie", "hack vs legitimate")]
    PROBE = {"board_seed": ("probe Lall think-open+close", "mean"), "solo_design": ("linear per-step probe, last step", "last"),
             "spar_screen_team": ("linear per-step probe, last step", "last"), "spar_mixed_41": ("linear per-step probe, last step", "last"),
             "goodfire_reward_hack": ("probe think pair, layers 26+38+51", "mean"), "cheatbench": ("probe think pair, layers 26+38+51", "mean"),
             "evilgenie": ("probe think pair, layers 26+38+51", "mean")}
    T = ["| set | label | positive / negative | linear probe | readout | attention monitor | TF-IDF, agent's words | best cheap judge | reader interrogation |",
         "|---|---|---|---|---|---|---|---|---|"]
    beats = []
    for st, cut in MAIN:
        rows = [r for r in jc if r["set"] == st and r["cut"] == cut]
        full = max(rows, key=lambda r: int(r["scored"]))
        pos, neg = full["pos"], full["neg"]
        short = max(int(full["scored"]) - int(r["scored"]) for r in rows)
        ctx["main_shortfall"] = max(ctx.get("main_shortfall", 0), short)
        if st == "controls_600":
            pr = ctx["controls_solo_pca"]
            probe, readout = f3(pr["auroc"]), "mean"
            assert (pr["cheat"], pr["no_cheat"]) == (pos, neg), (pr["cheat"], pr["no_cheat"], pos, neg)
            led(probe, f"Table 3 row {st}/{cut} col linear probe", "hermeneutics/results/pca_agent/auroc.csv",
                "arm=solo,probe=pca,layer=all,position=both,pooling=agent,features=raw,scope=pooled -> auroc (the agents of the judges' solo cut)", raw_where=True)
        else:
            name, readout = PROBE[st]
            pr = jrow(st, cut, name)
            probe = f3(pr["auroc"])
            led(probe, f"Table 3 row {st}/{cut} col linear probe", src_j, f"set={st},cut={cut},scorer={name} -> auroc", raw_where=True)
        mon = jrow(st, cut, "attention monitor, last step")
        tf = jrow(st, cut, "TF-IDF agent's words") or jrow(st, cut, "TF-IDF agent's words so far, last step")
        js = [(float(jrow(st, cut, k)["auroc"]), lab) for k, lab in JUDGES]
        best = max(js)
        rd = jrow(st, cut, READER)
        cells = [probe, readout, f3(mon["auroc"]) if mon else "not run", f3(tf["auroc"]), f"{best[0]:.3f} ({SHORT[best[1]]})", f3(rd["auroc"])]
        for v, what in [(cells[2], "attention monitor"), (cells[3], "TF-IDF"), (cells[4], "best judge"), (cells[5], "reader interrogation")]:
            led(v, f"Table 3 row {st}/{cut} col {what}", src_j, f"set={st},cut={cut} -> auroc; best judge = max over the three judges", raw_where=True)
        T.append(f"| {SETNAME[st]} | {CUTNAME[cut]} | {pos} / {neg} | " + " | ".join(cells) + " |")
        internals = max(float(probe), float(mon["auroc"]) if mon else 0.0)
        beats.append((st, cut, internals > float(tf["auroc"]), internals > best[0], internals > float(rd["auroc"])))
    out["<<T_MAIN>>"] = "\n".join(T)
    ctx["main_beats"] = beats

    # ------------------------------------------------------------------ per-set judge tables (appendix)
    order = ["controls_600", "board_seed", "solo_design", "spar_screen_team", "spar_mixed_41", "goodfire_reward_hack", "cheatbench", "evilgenie"]
    JT = []
    for st in order:
        cuts = list(dict.fromkeys(r["cut"] for r in jc if r["set"] == st))
        L = [f"**Table {{N}}.** {SETNAME[st]}: every scorer on the same agents. AUROC [95% bootstrap over runs, rollouts or pairs]; within: { {'controls_600': 'the mean of the per-author AUROCs', 'board_seed': 'pairs inside one seed arm', 'solo_design': 'one stratum, equal to the AUROC', 'spar_screen_team': 'pairs inside one task', 'spar_mixed_41': 'pairs inside one task', 'goodfire_reward_hack': 'pairs inside one run directory', 'cheatbench': 'pairs inside one harness', 'evilgenie': 'pairs inside one author'}[st] }; judges' Yes rate and Cohen's kappa against the code label.", "",
             "| label | scorer | agents scored | positive / negative | AUROC [95% CI] | within | judge says Yes | kappa |", "|---|---|---|---|---|---|---|---|"]
        for cut in cuts:
            for r in [r for r in jc if r["set"] == st and r["cut"] == cut]:
                sc = r["scorer"].replace("judge: qwen/qwen3.8-27b", "reader interrogation (Qwen3.8-27B)")
                for k, lab in JUDGES:
                    sc = sc.replace(k, "judge: " + lab)
                sc = sc.replace("probe Lall think-open+close", "probe, think pair, agent mean").replace("probe think pair, layers 26+38+51", "probe, think pair, agent mean").replace("probe think pair, layer 26", "probe, think pair, layer 26")
                wk = "within_mean" if st == "controls_600" else "within"   # as judges_all RESULTS.md prints it per set
                within = f3(r[wk]) if r[wk] not in ("", "nan") else "n/a"
                yes = f"{float(r['yes_rate']):.2f}" if r["yes_rate"] else ""
                kap = f"{float(r['kappa']):.2f}" if r["kappa"] else ""
                line = f"| {CUTNAME[cut]} | {sc} | {r['scored']} | {r['pos']} / {r['neg']} | {ci(r)} | {within} | {yes} | {kap} |"
                led(line, f"Appendix judge table {st}/{cut}/{r['scorer']}", src_j, "scored, pos, neg, auroc, ci, within_mean, yes_rate, kappa", raw_where=True)
                L.append(line)
        JT.append("\n".join(L))
    out["<<T_JUDGE_SETS>>"] = JT

    # reader interrogation agreement (judges_all RESULTS.md)
    t = P.table(P.JA, JR, "Reader interrogation: does the reader say what its states encode?")
    R = ["| set | label | agents | reader AUROC | probe AUROC | reader says Yes | probe says Yes | agreement | kappa |", "|---|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        d = dict(zip(t["header"], r))
        line = f"| {SETNAME[d['Set']]} | {CUTNAME[d['Cut']]} | {d['Agents']} | {d['Reader AUROC']} | {d['Probe AUROC']} | {d['Reader says Yes']} | {d['Probe says Yes']} | {d['Agreement']} | {d['Kappa']} |"
        led(line, "Appendix reader-interrogation table", P.where(t, r), "row as printed", raw_where=True)
        R.append(line)
    out["<<T_INTERROG>>"] = "\n".join(R)

    # tail cut
    t = P.table(P.JA, JR, "Does the tail cut hide the onset?")
    R = ["| set | label | judge | positives scored | onset known | onset in view | AUROC, all | onset in view | onset cut away |", "|---|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        d = dict(zip(t["header"], r))
        jn = dict((k.split(": ")[1], lab) for k, lab in JUDGES).get(d["Judge"], "reader (Qwen3.8-27B)")
        line = f"| {SETNAME[d['Set']]} | {CUTNAME[d['Cut']]} | {jn} | {d['Positives scored']} | {d['Onset known']} | {d['Onset in view']} | {d['AUROC, all']} | {d['AUROC, onset in view']} | {d['AUROC, onset cut away']} |"
        led(line, "Appendix tail-cut table", P.where(t, r), "row as printed", raw_where=True)
        R.append(line)
    out["<<T_TAILCUT>>"] = "\n".join(R)

    # judge setup facts for the prose (read, ledgered; prose keeps them out of the body)
    t = P.table(P.JA, JR, "Setup", first_col="Part")
    prom = dict(zip(t["header"], next(r for r in t["rows"] if r[0] == "prompts")))["Choice"]
    led(prom, "Appendix E judge setup: prompts per judge and the share cut", P.where(t, None), "Setup row 'prompts'", raw_where=True)
    out["<<JUDGE_PROMPTS>>"] = prom.split(" per judge")[0]
    out["<<JUDGE_CUT_SHARE>>"] = prom.split("cut in ")[1].split(" of them")[0]

    # per-step judge prefixes (judges_all step table)
    t = P.table(P.JA, JR, "Detection delay: how soon does a judge flag cheating?", first_col="Set")
    out["_judge_step"] = [dict(zip(t["header"], r)) for r in t["rows"]]
    out["_judge_step_where"] = P.where(t, None)

    # ------------------------------------------------------------------ unseen authors (main)
    U = ["| set | split | positive / agents | linear probe | attention monitor (seeds 0, 1) | TF-IDF |", "|---|---|---|---|---|---|"]

    def attn_heldout(tag, cut_head, label, set_name):
        tb = P.table(P.TEAM, AP, tag, first_col="Model", last=cut_head)
        sub = [h for h in tb["heads"] if "agents scored per step" in h]
        g = lambda m: P.row(tb, Model=m)
        m0, m1 = g("attention monitor, binary head (seed 0)"), g("attention monitor, binary head (seed 1)")
        lin, tfr = g("linear per-step probe (team_probes)"), g("TF-IDF on the agent's words so far")
        a = lambda d: d["Agent AUROC (last)"].split(" [")[0]
        txt = P.show(P.TEAM, AP)
        sec = txt.split(f": {tag}\n", 1)[1].split(f"### {cut_head}", 1)[1]
        import re as _re
        m = _re.search(r"(\d+) agents scored per step \((\d+) reach the cut\)", sec)
        n_ag = m.group(2) + " / " + m.group(1)
        line = f"| {set_name} | {label} | {n_ag} | {a(lin)} | {a(m0)}, {a(m1)} | {a(tfr)} |"
        led(line, f"Table 4 row {set_name}/{label}", f"{AP}@{P.TEAM} section {tag} > {cut_head}", "agent AUROC (last) of the linear per-step probe, the monitor seeds 0 and 1, and TF-IDF on the agent's words so far; counts from the section's 'agents scored per step' line", raw_where=True)
        return line

    # controls: matrix B.II (author table) against the whole-text TF-IDF B.II (lexical table)
    ta = P.table(P.CN, CP, "2. Does the type B probe read author identity?")
    tl = P.table(P.CN, CP, "3. Lexical-only reader null")
    ra = P.row(ta, Label="cheating", Scheme="alone: controls")
    rl = P.row(tl, Label="cheating", Scheme="alone: controls", **{"Probe type": "B.II"})
    line = f"| Controls | leave one author out, mean over five | {ra['Positive']} / {ra['Agents']} | {ra['B.II mean raw'].split(' [')[0]} | not run | {rl['Whole-text TF-IDF'].split(' [')[0]} |"
    led(line, "Table 4 row Controls B.II", f"{CP}@{P.CN}", "author table 'alone: controls' B.II mean raw; lexical table B.II whole-text TF-IDF", raw_where=True)
    U.append(line)
    U.append(attn_heldout("controls_600_think_heldout_author", "cheating (the set's label: from tried or opened by task)", "held-out author, pooled", "Controls"))
    U.append(attn_heldout("solo_design_think_heldout_author", "cheating used or above", "held-out author, pooled", "Design runs"))
    U.append(attn_heldout("spar_screen_team_think_heldout_author", "cheating (the set's label: from tried or opened by task)", "held-out author, pooled", "SPAR screen teams"))
    U.append(attn_heldout("spar_mixed_41_think_heldout_author", "cheating (the set's label: from tried or opened by task)", "held-out author, pooled", "SPAR mixed teams"))
    # external held-out author: within the held-out author (Goodfire: within run dir)
    for set_name, head, wcol, tcol_note in [("Goodfire", "1b. Held-out author", "Within run dir AUROC [95% CI]", "within run directory"),
                                            ("CheatBench", "CheatBench, cheating or attempt vs benign: held-out author", "Within held-out author AUROC [95% CI]", "within the held-out author"),
                                            ("EvilGenie", "EvilGenie, hack vs legitimate: held-out author", "Within held-out author AUROC [95% CI]", "within the held-out author")]:
        tb = P.table(P.EXT, EP, head, first_col="Features")
        pr = P.row(tb, Features="think pair, layers 26+38+51")
        tf = P.row(tb, Features="TF-IDF agent's words (rendered text)")
        line = f"| {set_name} | held-out author, {tcol_note} | {pr['Positive']} / {pr['Rollouts']} | {pr[wcol].split(' [')[0]} | not run | {tf[wcol].split(' [')[0]} |"
        led(line, f"Table 4 row {set_name} held-out author", f"{EP}@{P.EXT} > {head}", f"column {wcol}", raw_where=True)
        U.append(line)
    # transfer both ways (external_probes Question 3) and controls to the team workers (team_probes probe 6)
    tq = P.table(P.EXT, EP, "Trained on controls solo (373 agents, 193 cheat)", first_col="Target")
    r = P.row(tq, Target="Goodfire judge sets (verdict yes vs no)", Group="all")
    line = f"| Controls to Goodfire | train on the controls' solo arm, score Goodfire | {r['Positive']} / {r['Rows']} | {r['probe layers 26+38+51'].split(' [')[0]} | not run | {r['TF-IDF'].split(' [')[0]} |"
    led(line, "Table 4 row transfer controls to Goodfire", f"{EP}@{P.EXT} > Question 3, trained on controls solo", "Goodfire judge sets, all", raw_where=True)
    U.append(line)
    tq = P.table(P.EXT, EP, "Trained on Goodfire judge sets (2833 rollouts, 2083 yes)", first_col="Target")
    r = P.row(tq, Target="controls solo", Group="all")
    line = f"| Goodfire to controls | train on Goodfire, score the controls' solo arm | {r['Positive']} / {r['Rows']} | {r['probe layers 26+38+51'].split(' [')[0]} | not run | {r['TF-IDF'].split(' [')[0]} |"
    led(line, "Table 4 row transfer Goodfire to controls", f"{EP}@{P.EXT} > Question 3, trained on Goodfire", "controls solo, all", raw_where=True)
    U.append(line)
    tq = P.table(P.TEAM, TP, "Probe 6", first_col="Features", last="Trained on controls solo")
    pr = P.row(tq, Features="probe Lall think-open+close", **{"Scored on": "team workers, cheating used or above"})
    tf = P.row(tq, Features="TF-IDF agent's words", **{"Scored on": "team workers, cheating used or above"})
    line = f"| Controls to board-seed teams | train on the controls' solo arm, score the workers | {pr['Positive']} / {pr['Rows']} | {pr['AUROC [95% CI]'].split(' [')[0]} | not run | {tf['AUROC [95% CI]'].split(' [')[0]} |"
    led(line, "Table 4 row transfer controls to board-seed workers", f"{TP}@{P.TEAM} > Probe 6 > Trained on controls solo", "team workers, cheating used or above", raw_where=True)
    U.append(line)
    out["<<T_UNSEEN>>"] = "\n".join(U)

    # ------------------------------------------------------------------ per step (main): board seed and Goodfire
    S = ["| set and label | features | turn AUROC | delay, turns | detected |", "|---|---|---|---|---|"]
    for lab, sec in [("Board-seed teams, cheating used", "5a cheating used, onset = first write of planted content, setting B"),
                     ("Board-seed teams, collusion crossed", "5d collusion crossed, onset proxy, setting B")]:
        tb = P.table(P.TEAM, TP, sec, first_col="Features")
        for feat, nm in [("probe Lall think-open+close", "linear probe, think pair"), ("probe Lall all seven tokens (boundary 1-5 + think pair)", "linear probe, seven tokens"),
                         ("TF-IDF agent's words so far", "TF-IDF, agent's words so far"), ("turn position (index, fraction)", "turn position")]:
            r = P.row(tb, Features=feat)
            det = r["Detected / reaching"].replace(" / ", " / ")
            line = f"| {lab} | {nm} | {r['Turn AUROC [95% CI]'].split(' [')[0]} | {r['Delay median [IQR]'].split(' [')[0]} | {det} |"
            led(line, f"Table 5 row {lab}/{nm}", P.where(tb, r), "Turn AUROC, Delay median, Detected / reaching", raw_where=True)
            S.append(line)
    for d_name, head in [("A", "Definition A"), ("B", "Definition B")]:
        tb = P.table(P.EXT, EP, "Question 2", head, first_col="Features")
        lab = f"Goodfire, flagged passage (definition {d_name})"
        for feat, nm in [("linear probe, think pair, layers 26+38+51", "linear probe, think pair"),
                         ("attention monitor (layer-38 think pair, trained on definition B)", "attention monitor"),
                         ("TF-IDF agent's words so far", "TF-IDF, agent's words so far"), ("turn index and fraction of the rollout", "turn position")]:
            r = P.row(tb, Features=feat)
            det = r["Yes rollouts detected"].replace(" of ", " / ")
            line = f"| {lab} | {nm} | {r['Step AUROC [95% CI]'].split(' [')[0]} | {r['Delay, turns: median [q25, q75]'].split(' [')[0]} | {det} |"
            led(line, f"Table 5 row {lab}/{nm}", P.where(tb, r), "Step AUROC, Delay median, Yes rollouts detected", raw_where=True)
            S.append(line)
    for d in out["_judge_step"]:
        if d["Alarm"].startswith("score at 5% FPR") and d["Judge"].startswith("deepseek"):
            lab = {"board_seed": "Board-seed teams, cheating used", "goodfire_reward_hack": "Goodfire, verdict"}[d["Set"]]
            pn = d["Agents (pos / neg)"]
            delay = d["Delay, turns: median [q25, q75]"].split(" [")[0]
            delay = delay if delay[:1].isdigit() else "none"
            line = f"| {lab} (subsample {pn}) | cheap judge (DeepSeek V4.1 Flash) on prefixes | not computed | {delay} | {d['Detected'].replace(' of ', ' / ')} |"
            led(line, f"Table 5 row judge prefixes {d['Set']}", out["_judge_step_where"], "deepseek row at 5% FPR: Detected, Delay", raw_where=True)
            S.append(line)
    out["<<T_STEP_ALL>>"] = "\n".join(S)

    # ------------------------------------------------------------------ prompt-only control (main)
    tp = [t for t in P.md_tables(P.CN, CP) if t["header"][:2] == ["Label", "Set"] and "Prompt-only AUROC [95% CI]" in t["header"]][0]
    PR = ["| label | set | positive / agents | prompt-only probe | null 95th | task and arm identity | full probe | TF-IDF, whole text | share of the probe's margin |", "|---|---|---|---|---|---|---|---|---|"]
    LN = {"cheating": "cheating", "coordination posted": "coordination", "collusion crossed": "collusion"}
    SN = dict(SETNAME, spar_exploration_single="SPAR exploration", spar_screen_solo="SPAR screen solo")
    for r in tp["rows"]:
        d = dict(zip(tp["header"], r))
        line = (f"| {LN[d['Label']]} | {SN[d['Set']]} | {d['Positive']} / {d['Agents']} | {d['Prompt-only AUROC [95% CI]'].split(' [')[0]} | "
                f"{d['Null mean / 95th'].split(' / ')[1]} | {d['Task + arm identity']} | {d['Full probe (think pair)'].split(' [')[0]} | {d['TF-IDF rendered text']} | {d['Share of full margin']} |")
        led(line, f"Table 7 row {d['Label']}/{d['Set']}", P.where(tp, r), "Prompt-only AUROC, null 95th, task+arm identity, full probe (think pair), TF-IDF, share", raw_where=True)
        PR.append(line)
    out["<<T_PROMPT>>"] = "\n".join(PR)
    shares = {(dict(zip(tp["header"], r))["Set"], dict(zip(tp["header"], r))["Label"]): float(dict(zip(tp["header"], r))["Share of full margin"]) for r in tp["rows"]}
    ctx["prompt_shares"] = shares

    # within-condition (appendix): conditions with a refit
    tw = [t for t in P.md_tables(P.CN, CP) if t["header"][:3] == ["Label", "Set", "Condition"]][0]
    W = ["| label | set | condition | positive / agents | restricted: prompt-only | restricted: full probe | refit: prompt-only | refit null 95th | refit: full probe |", "|---|---|---|---|---|---|---|---|---|"]
    for r in tw["rows"]:
        d = dict(zip(tw["header"], r))
        if not d["Refit: prompt-only [95% CI]"][:1].isdigit():
            continue
        line = (f"| {LN[d['Label']]} | {SN[d['Set']]} | {d['Value'].replace(chr(92) + '|', ', ').replace('|', ', ')} | {d['Positive']} / {d['Agents']} | {d['Restricted: prompt-only']} | {d['Restricted: full']} | "
                f"{d['Refit: prompt-only [95% CI]'].split(' [')[0]} | {d['Refit prompt-only null 95th']} | {d['Refit: full']} |")
        led(line, f"Appendix within-condition row {d['Label']}/{d['Set']}/{d['Value']}", P.where(tw, r), "restricted and refit columns", raw_where=True)
        W.append(line)
    out["<<T_WITHIN>>"] = "\n".join(W)

    # author identity and LEACE (main)
    AU = ["| label | training scheme | authors | positive / agents | author probe: raw | after erasure (chance) | B.I: raw, author erased, all three erased | B.II mean: raw, author erased, all three erased |",
          "|---|---|---|---|---|---|---|---|"]
    SCH = {"alone: controls": "controls", "alone: solo_design": "design runs", "alone: spar_team": "SPAR screen teams", "alone: spar_mixed": "SPAR mixed teams",
           "alone: goodfire": "Goodfire", "ours": "all our sets", "all": "all sets"}
    for r in ta["rows"]:
        d = dict(zip(ta["header"], r))
        s0 = lambda k: d[k].split(" [")[0]
        line = (f"| {LN[d['Label']]} | {SCH[d['Scheme']]} | {d['Authors']} | {d['Positive']} / {d['Agents']} | {d['Author probe: raw']} | {d['Author: after author erasure']} ({d['Chance']}) | "
                f"{s0('B.I raw')}, {s0('B.I author-erased')}, {s0('B.I author+task+condition-erased')} | {s0('B.II mean raw')}, {s0('B.II mean author-erased')}, {s0('B.II mean author+task+condition-erased')} |")
        led(line, f"Table 8 row {d['Label']}/{d['Scheme']}", P.where(ta, r), "author probe and B.I / B.II columns", raw_where=True)
        AU.append(line)
    out["<<T_AUTHOR>>"] = "\n".join(AU)
    ctx["author_rows"] = {(dict(zip(ta["header"], r))["Label"], dict(zip(ta["header"], r))["Scheme"]): dict(zip(ta["header"], r)) for r in ta["rows"]}

    # lexical-only null (appendix): the alone and pooled schemes
    LX = ["| label | scheme | probe type | positive / agents | think-window TF-IDF | agents with a think window | whole-text TF-IDF | full probe |", "|---|---|---|---|---|---|---|---|"]
    for r in tl["rows"]:
        d = dict(zip(tl["header"], r))
        if not (d["Scheme"].startswith("alone") or d["Scheme"] in ("ours", "external", "all")):
            continue
        line = (f"| {LN[d['Label']]} | {d['Scheme'].replace('alone: ', '')} | {d['Probe type']} | {d['Positive']} / {d['Agents']} | {d['Think-window TF-IDF'].split(' [')[0]} | "
                f"{d['Agents with a think-window score']} | {d['Whole-text TF-IDF'].split(' [')[0]} | {d['Full probe (think pair)'].split(' [')[0]} |")
        led(line, f"Appendix lexical-null row {d['Label']}/{d['Scheme']}/{d['Probe type']}", P.where(tl, r), "think-window and whole-text TF-IDF, full probe", raw_where=True)
        LX.append(line)
    out["<<T_LEXICAL>>"] = "\n".join(LX)

    # null audit (appendix)
    tn = [t for t in P.md_tables(P.CN, CP) if t["header"][0] == "Results dir" and t["header"][1] == "Rows with a null"][0]
    NA = ["| result set | rows with a null | rows without | rows without, null computed in the audit |", "|---|---|---|---|"]
    NAME = {"attn_probe": "attention monitor", "external_probes": "external sets", "matrix_probes": "probe matrix", "team_probes": "team-run probes"}
    for r in tn["rows"]:
        d = dict(zip(tn["header"], r))
        line = f"| {NAME[d['Results dir']]} | {d['Rows with a null']} | {d['Rows without']} | {d['Rows without, null computed here']} |"
        led(line, "Appendix null audit", P.where(tn, r), "row as printed", raw_where=True)
        NA.append(line)
    out["<<T_NULLAUDIT>>"] = "\n".join(NA)
    total = P.show(P.CN, CP).split("Total still without a null: ", 1)[1].split(" rows", 1)[0]
    out["<<NULL_LEFT>>"] = total
    led(total, "Appendix null audit: rows still without a null", f"{CP}@{P.CN}", "'Total still without a null'", raw_where=True)

    # ------------------------------------------------------------------ external sets (appendix): Goodfire per run dir, CheatBench harness confound
    tb = P.table(P.EXT, EP, "1a. Five folds", first_col="Run dir")
    G = ["| run directory | yes / no | probe, layers 26, 38, 51 | probe, layer 26 | TF-IDF | metadata |", "|---|---|---|---|---|---|"]
    for r in tb["rows"]:
        d = dict(zip(tb["header"], r))
        line = f"| {d['Run dir']} | {d['Yes']} / {d['No']} | {d['probe layers 26+38+51']} | {d['probe layer 26']} | {d['TF-IDF']} | {d['metadata']} |"
        led(line, "Appendix Goodfire per run directory", P.where(tb, r), "row as printed", raw_where=True)
        G.append(line)
    out["<<T_GF_RUNDIR>>"] = "\n".join(G)
    tb = P.table(P.EXT, EP, "1a. Five folds", first_col="Features", nth=0)  # the main five-fold table; the sensitivity table follows it
    G = ["| features | yes / no | pooled AUROC | pooled null 95th | within run directory | within null 95th |", "|---|---|---|---|---|---|"]
    for r in tb["rows"]:
        d = dict(zip(tb["header"], r))
        nl = lambda k: d[k].split(" / ")[1] if d[k] else "not run"
        line = f"| {d['Features']} | {d['Positive']} / {d['Negative']} | {d['Pooled AUROC [95% CI]']} | {nl('Pooled null mean / 95th')} | {d['Within run dir AUROC [95% CI]']} | {nl('Within null mean / 95th')} |"
        led(line, "Appendix Goodfire five folds", P.where(tb, r), "row as printed", raw_where=True)
        G.append(line)
    out["<<T_GF_FOLDS>>"] = "\n".join(G)
    tb = P.table(P.EXT, EP, "Question 0", first_col="Harness")
    C = ["| harness | rollouts | cheat / benign | AUROC | mean score, cheat | mean score, benign |", "|---|---|---|---|---|---|"]
    for r in tb["rows"]:
        d = dict(zip(tb["header"], r))
        line = f"| {d['Harness']} | {d['Rollouts']} | {d['Cheat']} / {d['Benign']} | {d['AUROC']} | {d['Mean score, cheat'] or 'n/a'} | {d['Mean score, benign'] or 'n/a'} |"
        led(line, "Appendix CheatBench harness confound", P.where(tb, r), "row as printed", raw_where=True)
        C.append(line)
    tb = P.table(P.EXT, EP, "Question 0", first_col="Pairs")
    C2 = ["| pairs | count | AUROC over these pairs |", "|---|---|---|"]
    for r in tb["rows"]:
        d = dict(zip(tb["header"], r))
        line = f"| {d['Pairs']} | {d['Count']} | {d['AUROC over these pairs']} |"
        led(line, "Appendix CheatBench pairs", P.where(tb, r), "row as printed", raw_where=True)
        C2.append(line)
    out["<<T_CB_HARNESS>>"] = "\n".join(C)
    out["<<T_CB_PAIRS>>"] = "\n".join(C2)
    tb = P.table(P.EXT, EP, "EvilGenie, hack vs legitimate: five folds", first_col="Features")
    E = ["| features | hack / legitimate | pooled | within author | within problem | within author and problem |", "|---|---|---|---|---|---|"]
    for r in tb["rows"]:
        d = dict(zip(tb["header"], r))
        line = (f"| {d['Features']} | {d['Positive']} / {d['Negative']} | {d['Pooled AUROC [95% CI]'].split(' [')[0]} | {d['Within author AUROC [95% CI]'].split(' [')[0]} | "
                f"{d['Within problem AUROC [95% CI]'].split(' [')[0]} | {d['Within author and problem AUROC [95% CI]'].split(' [')[0]} |")
        led(line, "Appendix EvilGenie five folds", P.where(tb, r), "row as printed", raw_where=True)
        E.append(line)
    out["<<T_EG>>"] = "\n".join(E)

    # ------------------------------------------------------------------ team-run agent probes per set (appendix)
    A = ["| set | label | positive / negative | think pair | seven tokens | five boundary tokens | TF-IDF, agent's words | TF-IDF, whole text | metadata | null 95th, seven tokens |", "|---|---|---|---|---|---|---|---|---|---|"]
    SETS = [("Board-seed teams", None, [("cheating, used", "1a cheating: used or above, setting B"), ("collusion, crossed", "2 collusion: exploit crossed or above vs none, setting B"),
                                        ("coordination, posted", "3a coordination: posted or above, setting B"), ("coordination, discussed", "3b coordination: discussed or above, setting B")]),
            ("Design runs", "solo_design, pooled", [("cheating, used", "1a cheating: used or above, setting B"), ("coordination, read the board", "3r coordination: read the board, setting B"),
                                                    ("coordination, posted", "3a coordination: posted or above, setting B")]),
            ("SPAR screen teams", "spar_screen_team, all", [("cheating, used", "1a cheating: used or above, setting B"), ("collusion, crossed", "2 collusion: exploit crossed or above vs none, setting B"),
                                                            ("coordination, posted", "3a coordination: posted or above, setting B")]),
            ("SPAR mixed teams", "spar_mixed_41, all", [("cheating, used", "1a cheating: used or above, setting B"), ("collusion, crossed", "2 collusion: exploit crossed or above vs none, setting B")])]
    for set_name, prefix, labels in SETS:
        for lab, head in labels:
            h = head if prefix is None else f"{prefix}: {head}"
            cands = [t for t in P.md_tables(P.TEAM, TP) if t["heads"][-1].startswith(h) and t["header"][0] == "Features"]
            assert len(cands) == 1, (h, len(cands))
            tb = cands[0]
            g = lambda f: P.row(tb, Features=f)
            s7 = g("probe Lall all seven tokens (boundary 1-5 + think pair)")
            pos = s7["Positive"]; neg = s7["Negative"]
            a = lambda f: g(f)["AUROC [95% CI]"].split(" [")[0]
            line = (f"| {set_name} | {lab} | {pos} / {neg} | {a('probe Lall think-open+close')} | {a('probe Lall all seven tokens (boundary 1-5 + think pair)')} | "
                    f"{a('probe Lall boundary 1-5 concatenated')} | {a(chr(84) + 'F-IDF agent' + chr(39) + 's words')} | {a('TF-IDF whole text')} | {a('metadata (arm, seed, turn budget)')} | {s7['Null 95th pct']} |")
            led(line, f"Appendix team agent probes {set_name}/{lab}", P.where(tb, None), "AUROC column per feature row; null 95th of the seven-token row", raw_where=True)
            A.append(line)
    out["<<T_TEAM_SETS>>"] = "\n".join(A)

    # ------------------------------------------------------------------ attention-monitor nulls on the SPAR and design sets (for Table D8)
    tn = P.table(P.TEAM, AP, "Permutation nulls on the SPAR and solo sets", first_col="Run")
    ctx["attn_nulls"] = {(d["Run"], d["Cut"], d["Model"], d["AUROC"]): d for d in (dict(zip(tn["header"], r)) for r in tn["rows"])}
    ctx["attn_nulls_where"] = P.where(tn, None)

    # three seeds (appendix)
    tb = P.table(P.TEAM, AP, "Three seeds of the headline rows", first_col="Run")
    TS = ["| run | label | step AUROC, seeds 0, 1, 2 | agent AUROC (last), seeds 0, 1, 2 |", "|---|---|---|---|"]
    RN = {"board_seed_think": "board-seed teams, think pair", "board_seed_seven": "board-seed teams, seven tokens", "controls_600_think": "controls, think pair", "controls_600_seven": "controls, seven tokens"}
    for r in tb["rows"]:
        d = dict(zip(tb["header"], r))
        line = f"| {RN[d['Run']]} | {CUTNAME.get(d['Cut'], d['Cut'])} | {d['Step AUROC, seeds 0 / 1 / 2'].replace(' / ', ', ')} | {d['Agent AUROC (last), seeds 0 / 1 / 2'].replace(' / ', ', ')} |"
        led(line, "Appendix three seeds", P.where(tb, r), "row as printed", raw_where=True)
        TS.append(line)
    out["<<T_SEEDS>>"] = "\n".join(TS)

    # ------------------------------------------------------------------ the sets (main Table 1)
    td = P.table(P.TEAM, TP, first_col="Set", last="Data")
    lex = P.csv_rows(P.CN, "hermeneutics/results/controls_nulls/lexical_rows.csv")
    auth = {}
    for r in lex:
        if r["scheme"].startswith("alone: ") and r["unit"] not in ("pooled", "mean over authors"):
            auth.setdefault(r["scheme"][7:], set()).add(r["unit"])
    KEY = {"controls_600": "controls", "board_seed": "board_seed", "solo_design": "solo_design", "spar_screen_team": "spar_team", "spar_mixed_41": "spar_mixed",
           "spar_exploration_single": "spar_expl", "spar_screen_solo": "spar_solo", "goodfire_reward_hack": "goodfire", "cheatbench": "cheatbench", "evilgenie": "evilgenie"}
    INFO = {"controls_600": ("ours, this work", "one agent per run, nine tasks with a planted exploit, two prompt arms", "cheating"),
            "board_seed": ("ours, this work", "three workers share a board; one seeded post", "six ladders"),
            "solo_design": ("ours, this work", "three lone agents per task with board tools", "six ladders"),
            "spar_screen_team": ("SPAR team, before this work", "three-agent teams, one author per team", "six ladders"),
            "spar_mixed_41": ("SPAR team, before this work", "teams of two worker authors, other-model regulators", "six ladders"),
            "spar_exploration_single": ("SPAR team, before this work", "single agents across prompt arms", "cheating"),
            "spar_screen_solo": ("SPAR team, before this work", "single agents, two prompt arms", "cheating"),
            "goodfire_reward_hack": ("external", "coding and tool-use rollouts", "judge verdict; flagged passages"),
            "cheatbench": ("external", "three agent harnesses", "human label"),
            "evilgenie": ("external", "matched hack and legitimate pairs", "dataset label")}
    SN = dict(SETNAME, spar_exploration_single="SPAR exploration", spar_screen_solo="SPAR screen solo")
    TS = ["| set | made by | design | authors | agents | runs or source directories | labels |", "|---|---|---|---|---|---|---|"]
    for k in ["controls_600", "board_seed", "solo_design", "spar_screen_team", "spar_mixed_41", "spar_exploration_single", "spar_screen_solo",
              "goodfire_reward_hack", "cheatbench", "evilgenie"]:
        d = P.row(td, Set=k)
        made, design, labels = INFO[k]
        n_auth = len(auth[KEY[k]])
        line = f"| {SN[k]} | {made} | {design} | {n_auth} | {d['Agents on the Hub']} | {d['Runs on the Hub']} | {labels} |"
        led(line, f"Table 1 row {k}", P.where(td, None) + "; authors: distinct units of controls_nulls/lexical_rows.csv@" + P.CN, "Agents on the Hub, Runs on the Hub; authors counted", raw_where=True)
        TS.append(line)
    out["<<T_SETS>>"] = "\n".join(TS)
    AN = ["| set | authors |", "|---|---|"]
    for k in ["controls_600", "board_seed", "solo_design", "spar_screen_team", "spar_mixed_41", "spar_exploration_single", "spar_screen_solo",
              "goodfire_reward_hack", "cheatbench", "evilgenie"]:
        AN.append(f"| {SN[k]} | {', '.join(sorted(auth[KEY[k]]))} |")
    out["<<T_SET_AUTHORS>>"] = "\n".join(AN)

    # truncation count (34) from the truncated-transcript table
    tb = [t for t in P.md_tables(P.TEAM, TP) if "Truncated SPAR transcripts" in " ".join(t["heads"]) and "With truncated_at" in t["header"]][0]
    n = sum(int(dict(zip(tb["header"], r))["With truncated_at"]) for r in tb["rows"])
    out["<<N_TRUNC>>"] = str(n)
    led(n, "Appendix E reader passes: truncated SPAR agents", P.where(tb, None), "sum of 'With truncated_at' over the four SPAR sets", raw_where=True)
    return out
