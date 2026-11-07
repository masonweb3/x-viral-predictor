#!/usr/bin/env python3
"""X 爆款预测器：发帖前给几版稿打分，发后 12 小时再看一眼。只用标准库，Python >= 3.9。

  python3 xviral.py pull --authors authors.txt [--pages N]   # 拉同赛道作者的时间线
  python3 xviral.py train [--yes]                            # 打特征并训练，存 xviral-data/model.json
  python3 xviral.py score 稿1.txt [稿2.txt ...]              # 给每版稿打分并排序
  python3 xviral.py early <帖子链接>                          # 发后 12 小时起，推算最终浏览
  python3 xviral.py selftest                                 # 离线自检

环境变量：JEV_API_KEY（或当前目录 .env）；XVIRAL_DIR 改数据目录（默认 ./xviral-data）。
"""
import bisect
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JEV, API = "jev-1.14.0", "https://api.typesafe.ai/v1/systemone"
DIR = Path(os.environ.get("XVIRAL_DIR", "xviral-data"))
POSTS, FEATS = DIR / "posts.jsonl", DIR / "feats.jsonl"
BUNDLED = Path(__file__).resolve().with_name("model.json")
LAM, VIRAL = 30.0, math.log10(3)
CAL = [0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45]  # 校准表边界（log10 倍）：1.00 / 1.26 / 1.58 / 2.00 / 2.24 / 2.51 / 2.82 倍
# 版本对比：和最高版的预测差（log10 倍）→ 说法
GAPS = ((0.15, "和最高版分不开"), (0.25, "最高版更好，同事件帖对约 6 成挑对"), (math.inf, "最高版明显更好，约 3/4"))
CJK = re.compile(r"[一-鿿]")
XURL = re.compile(r"https?://(?:x|twitter)\.com/(\w+)/status/(\d+)")

READER = ("The readers are Chinese-speaking X (Twitter) users, mostly in mainland China, who follow accounts about "
          "overseas payment cards (U cards), Hong Kong / overseas bank accounts, eSIMs and overseas phone numbers, AI tools "
          "and subscriptions, global trending gossip, and edgy or crude humour. `post` is one X post in Chinese.")


def S(q, levels):
    return {"type": "score", "instructions": q + " " + READER, "criteria": levels}


def N(q, yes, no):
    return {"type": "noul", "instructions": q + " " + READER, "criteria": {"true": yes, "false": no}}


Q = {
    "hook": S("How strongly does the first line of `post` make this reader stop scrolling?", [
        "Greeting, filler, or vague mood; nothing concrete", "Names a topic but gives no reason to read on",
        "Names a concrete product or event with a mild hook", "A concrete surprise, figure, benefit or problem in the first line",
        "The first line alone delivers a striking, specific payoff or shock this reader cares about"]),
    "concrete": S("How many concrete specifics does `post` give: figures, prices, dates, product or bank names, steps?", [
        "None; entirely vague", "One specific", "A few specifics", "Many specifics throughout", "Dense with exact figures, names and steps"]),
    "news": S("How much does `post` report something that just changed: a launch, closure, rule change, price change, new offer or incident?", [
        "Evergreen or personal chatter; nothing new", "Mentions a change in passing", "Reports a change, but it is minor or old",
        "Reports a fresh, clear change", "Breaking change with immediate consequences for this reader"]),
    "firsthand": S("How much is `post` based on the author's own first-hand experience (they applied, paid, tested, received)?", [
        "No personal experience; relays others or general talk", "Personal opinion only", "Some personal experience mixed with relayed info",
        "Mainly the author's own test or result", "A detailed first-hand record with the author's own figures or screenshots described"]),
    "actionable": S("How much can this reader act on `post` today (apply, buy, claim, avoid, configure)?", [
        "Nothing to act on", "Vague direction only", "An action is implied but details are missing",
        "Clear action with most details", "Clear, complete action this reader can take right now"]),
    "money": S("How strongly does `post` promise money gained or saved for the reader: free, cashback, discount, airdrop, cheaper route?", [
        "No money angle", "Weak or indirect money angle", "Some saving or gain mentioned", "A clear, sizeable gain or saving", "A large or free gain that is the core of the post"]),
    "urgency": S("How time-sensitive is `post`: deadline, limited quota, 'closing soon', act-now?", [
        "Not time-sensitive", "Slight timeliness", "A time window is mentioned", "Clear deadline or limited quota", "Act within hours or lose it"]),
    "emotion": S("How strong is the emotion in `post` (surprise, excitement, anger, frustration, fear)?", [
        "Flat and neutral", "Mild tone", "Noticeable emotion", "Strong emotion", "Intense, dominant emotion"]),
    "controversy": S("How likely is `post` to provoke disagreement, debate or strong replies?", [
        "Uncontroversial", "Slightly opinionated", "Takes a side on something people disagree about", "Provocative claim or criticism", "Openly inflammatory or polarising"]),
    "humor": S("How humorous or meme-like is `post`?", ["Not at all", "Slightly playful", "Some wit", "Clearly funny", "A joke or meme is the point"]),
    "save_worthy": S("How likely is this reader to bookmark `post` to use later (reference list, steps, checklist, directory)?", [
        "Nothing worth keeping", "Low reference value", "Some reusable info", "A useful reference", "A must-keep reference or complete guide"]),
    "share_worthy": S("How likely is this reader to forward `post` to a friend or group?", [
        "Not shareable", "Low", "Some chance", "Likely", "Very likely; it is news or value their friends need"]),
    "broad": S("How broad is the audience for `post`?", [
        "Only the author or a tiny niche", "A small niche", "Readers of this niche generally", "Most Chinese-speaking tech or money readers", "Almost any Chinese-speaking X user"]),
    "mainland": S("How specifically does `post` address what mainland-China users can or cannot do (ID, KYC, cards, access)?", [
        "Not at all", "Indirectly", "Somewhat", "Clearly written for mainland users", "Entirely about a mainland-user blocker or opening"]),
    "risk": S("How much does `post` warn about loss, ban, freeze, scam or failure?", [
        "No warning", "Slight caution", "Mentions a risk", "Clear warning", "Urgent warning is the main point"]),
    "big_name": S("How prominent are the brands, people or events in `post` (OpenAI, Claude, Apple, Musk, Binance, a major bank)?", [
        "No known names", "Obscure names only", "Niche-known names", "Well-known brands", "Top global brands or headline events"]),
    "complete": S("Is `post` self-contained, or a teaser that only points elsewhere (link, article, reply, DM)?", [
        "Only a pointer or teaser", "Mostly a teaser", "Half content, half pointer", "Mostly self-contained", "Fully self-contained"]),
    "is_list": N("Is `post` a curated list, collection, directory or comparison of several products or options?",
                 "It lists or compares several items", "It is about one thing or is chatter"),
    "giveaway": N("Does `post` give away something or ask readers to comment, like, follow or repost to get something?",
                  "Giveaway or engagement request", "No giveaway and no engagement request"),
    "asks_readers": N("Does `post` ask readers a question or invite their opinions?", "Asks readers", "Does not ask readers"),
    "format": {"type": "choice", "instructions": "What is the format of `post`? " + READER, "criteria": {
        "news": "Reports news or a change", "tutorial": "Step-by-step how-to", "experience": "Author's own test or result",
        "opinion": "Opinion, rant or commentary", "showcase": "Shows something the author made", "list": "List or comparison",
        "giveaway": "Giveaway or promotion", "question": "Asks a question", "chat": "Casual chat or joke"}},
    "curiosity_gap": S("How much does `post` withhold the key point so the reader must open the image, video or replies to find out?", [
        "Nothing withheld; the text says it all", "Slight tease", "Some teasing", "Clear tease; the answer is elsewhere", "Pure bait: a riddle or 'guess what' with the answer hidden"]),
    "needs_media": N("Does the meaning of `post` depend on an attached image or video that the text does not describe?",
                     "The text only makes sense with the attached image or video", "The text stands on its own"),
    "taboo": S("How sensational or taboo is the subject of `post`: sex, gray market, dark web, VPN / bypassing the firewall, gambling, scandal, forbidden content?", [
        "Not at all", "A faint hint", "Touches a sensitive subject", "Clearly sensational or taboo", "Shocking; the taboo is the whole point"]),
    "relatable": S("How much does `post` describe an everyday situation or pain point that ordinary Chinese people share (holidays, salary, bills, work, prices)?", [
        "Not relatable to ordinary people", "Slightly", "Somewhat", "Clearly relatable to many", "Almost every reader has lived this"]),
    "followup": N("Does `post` continue or update something the author posted earlier that readers responded to (a series, 'as promised', 'since you liked it')?",
                  "It is a follow-up to an earlier post", "It is standalone"),
    "milestone": N("Is `post` about the author's own account milestone, thanks to followers, or self-promotion of the account?",
                   "Account milestone, thanks or self-promotion", "Not about the account itself"),
    "access_news": N("Does `post` report that a service, product or feature has just opened to, or been blocked for, mainland-China users?",
                     "Reports a fresh opening or blocking for mainland users", "No such change"),
    "try_now": S("How much does `post` offer something free or cheap that anyone can try right now with little effort (a site, tool, eSIM, trial)?", [
        "Nothing to try", "Something to try but costly or hard", "Try-able with moderate effort", "Free or cheap and easy to try", "Free, instant, and anyone can do it in a minute"]),
    "jargon": S("How much insider jargon or background knowledge does a newcomer need to understand `post`?", [
        "None; anyone understands", "A little", "Some niche terms", "Mostly insider talk", "Only insiders can follow it"]),
    "diary": S("How much is `post` about the author's own small personal situation that mainly matters to the author (my card arrived, my first card, my plan)?", [
        "Not about the author at all", "Mentions the author in passing", "Half personal, half general", "Mostly the author's own situation", "Entirely a personal diary entry"]),
    "punchline": S("How much does `post` end on a witty contrast, twist or punchline?", [
        "No punchline", "Faint", "Some twist", "Clear punchline", "The whole post builds to a sharp punchline"]),
    "real_question": N("Is the whole of `post` one genuine question readers can answer from their own experience?",
                       "Yes, it is a real question to readers", "No"),
}
NAMES = {"hook": "首句抓人", "concrete": "具体数字和名字", "news": "刚发生的变化", "firsthand": "亲测", "actionable": "能马上动手",
         "money": "钱", "urgency": "时效", "emotion": "情绪", "controversy": "会引起争论", "humor": "好笑", "save_worthy": "会收藏",
         "share_worthy": "会转给朋友", "broad": "受众广", "mainland": "面向大陆用户", "risk": "风险提醒", "big_name": "知名品牌或人物",
         "complete": "自成一体", "is_list": "清单合集", "giveaway": "福利或求互动", "asks_readers": "问读者",
         "format": "形式", "curiosity_gap": "卖关子", "needs_media": "要看图才懂", "taboo": "猎奇擦边",
         "relatable": "日常共鸣", "followup": "系列续集", "milestone": "涨粉感谢", "access_news": "对大陆开放或关门",
         "try_now": "免费能马上试", "jargon": "圈内黑话", "diary": "个人流水账", "punchline": "包袱", "real_question": "真实提问"}

ZH = {"news": "新闻", "tutorial": "教程", "experience": "亲测", "opinion": "观点", "showcase": "作品", "list": "清单",
      "giveaway": "福利", "question": "提问", "chat": "闲聊"}
COLS = [f"{k}={o}" if q["type"] == "choice" else k for k, q in Q.items() for o in (q["criteria"] if q["type"] == "choice" else [None])]


def load_env():
    f = Path(".env")
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "JEV_API_KEY" and "JEV_API_KEY" not in os.environ:
                os.environ[k.strip()] = v.strip().strip("\"'")


# ---------- 数学 ----------
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def ridge_cv(Z, y, groups, lam=LAM):
    """Z 每行末尾是常数 1（不罚）。返回 (全量权重, 按组留一的预测)。"""
    p, G, B = len(Z[0]), {}, {}
    for z, t, g in zip(Z, y, groups):
        Gg = G.setdefault(g, [[0.0] * p for _ in range(p)])
        Bg = B.setdefault(g, [0.0] * p)
        for i, zi in enumerate(z):
            if zi:
                Gg[i] = [a + zi * b for a, b in zip(Gg[i], z)]
                Bg[i] += zi * t
    Gt = [[sum(G[g][i][j] for g in G) for j in range(p)] for i in range(p)]
    Bt = [sum(B[g][i] for g in B) for i in range(p)]
    fit = lambda Gm, Bm: solve([[Gm[i][j] + (lam if i == j < p - 1 else 0.0) for j in range(p)] for i in range(p)], Bm)
    pred = [0.0] * len(y)
    for g in G:
        w = fit([[Gt[i][j] - G[g][i][j] for j in range(p)] for i in range(p)], [Bt[i] - B[g][i] for i in range(p)])
        for k, (z, gg) in enumerate(zip(Z, groups)):
            if gg == g:
                pred[k] = sum(a * b for a, b in zip(w, z))
    return fit(Gt, Bt), pred


# ---------- 语料 ----------
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "x-viral-predictor/1"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except (OSError, ValueError):
            if i == 2:
                raise
            time.sleep(2)


def fx_rows(results, now):
    """FxTwitter 时间线条目 → 帖子行；回复、转推不收（Article 标记留着，训练时再滤）。"""
    for s in results:
        if s.get("type") != "status" or s.get("reposted_by") or s.get("replying_to"):
            continue
        a = s.get("author") or {}
        yield {"id": s["id"], "user": a.get("screen_name"), "ts": s.get("created_timestamp"), "views": s.get("views"),
               "media": [m.get("type") for m in (s.get("media") or {}).get("all", [])], "quote": bool(s.get("quote")),
               "article": bool(s.get("article")), "text": s.get("text") or "", "fetched": now}


def timeline(user, pages, cutoff=0):
    """作者时间线 ≤pages 页，翻到早于 cutoff 为止；间隔 ≥0.5 秒。"""
    cursor, out = "", []
    for _ in range(pages):
        url = f"https://api.fxtwitter.com/2/profile/{user}/statuses?count=20" + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
        try:
            d = get(url)
        except (OSError, ValueError) as e:
            print(f"  {user}：{e}", file=sys.stderr)
            break
        finally:
            time.sleep(0.5)
        res = [s for s in d.get("results") or [] if isinstance(s, dict)]
        out += list(fx_rows(res, time.time()))
        cursor = (d.get("cursor") or {}).get("bottom")
        if not res or not cursor or (res[-1].get("created_timestamp") or 0) < cutoff:
            break
    return out


def load_posts():
    return {r["id"]: r for r in map(json.loads, POSTS.open())} if POSTS.exists() else {}


def pull(authors_file, pages=20):
    users = [u.strip().lstrip("@") for u in Path(authors_file).read_text().splitlines() if u.strip() and not u.startswith("#")]
    DIR.mkdir(parents=True, exist_ok=True)
    posts, cutoff = load_posts(), time.time() - 90 * 86400
    print(f"{len(users)} 位作者，每人最多 {pages} 页，最慢约 {max(1, round(len(users) * pages * 1.5 / 60))} 分钟")
    for u in users:
        rows = timeline(u, pages, cutoff)
        for r in rows:
            posts[r["id"]] = r
        print(f"  {u:<20} {len(rows)}")
    POSTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in posts.values()))
    print(f"语料共 {len(posts)} 条 → {POSTS}")


def labeled(posts):
    """合格帖 + lift = log10(浏览+10) − 同作者前后 14 天其他帖的中位（不足 5 条用作者全期中位）。
    只收发出 ≥48 小时、中文 ≥10 字、非 Article 的帖；同作者同文字留浏览最高一条；作者 ≥8 条才算。"""
    keep, seen = [], set()
    for r in sorted(posts.values(), key=lambda r: -(r["views"] or 0)):
        body = re.sub(r"https?://\S+", "", r["text"]).strip()
        key = (r["user"].lower(), re.sub(r"\s", "", body))
        if (r["views"] is None or r["article"] or r["ts"] is None or r["fetched"] - r["ts"] < 48 * 3600
                or len(CJK.findall(body)) < 10 or key in seen):
            continue
        seen.add(key)
        keep.append(dict(r))
    by = {}
    for r in keep:
        by.setdefault(r["user"].lower(), []).append(r)
    out = []
    for rs in by.values():
        if len(rs) < 8:
            continue
        lv = [math.log10(r["views"] + 10) for r in rs]
        med = statistics.median(lv)
        for r, v in zip(rs, lv):
            near = [w for o, w in zip(rs, lv) if o is not r and abs(o["ts"] - r["ts"]) <= 14 * 86400]
            r["lift"] = v - (statistics.median(near) if len(near) >= 5 else med)
            out.append(r)
    return out


# ---------- Jev ----------
def ask_raw(state, model=JEV):
    body = {"model": model, "state": state, "questions": Q}
    req = urllib.request.Request(API, json.dumps(body).encode(), {
        "Authorization": "Bearer " + os.environ["JEV_API_KEY"], "Content-Type": "application/json", "User-Agent": "x-viral-predictor/1"})
    for wait in (2, 6, None):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["answers"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and wait:
                time.sleep(wait)
                continue
            raise SystemExit(f"Jev {e.code}: {e.read().decode(errors='replace')[:300]}")
        except OSError:
            if not wait:
                raise
            time.sleep(wait)


def ask(text, quote=False, media=False, model=JEV):
    """一次请求问全部问题 → {问题: 分数 / 概率}。"""
    state = {"post": text, "post_type": "quote of another post" if quote else "original post", "has_image_or_video": media}
    return {k: a.get("probabilities") if a["type"] == "choice" else a.get("noul", a.get("score")) for k, a in ask_raw(state, model).items()}


def vec(ans):
    v = []
    for k, q in Q.items():
        a = ans.get(k)
        if q["type"] == "choice":
            v += [(a or {}).get(o, 0.0) for o in q["criteria"]]
        else:
            v.append(0.0 if a is None else a / (len(q["criteria"]) - 1) if q["type"] == "score" else a)
    return v


def safe_ask(r):
    try:
        return ask(r["text"], r["quote"], bool(r["media"]))
    except (SystemExit, OSError, ValueError, KeyError) as e:
        print(f"  Jev 跳过 {r['id']}：{e}", file=sys.stderr)


def load_feats():
    """{帖 id: 答案}，只取本版 Jev 问的行；追加到一半的末行跳过。"""
    rows = map(json.loads, (l for l in FEATS.open() if l.endswith("\n"))) if FEATS.exists() else ()
    return {j["id"]: j["a"] for j in rows if j.get("m") == JEV}


# ---------- 训练 ----------
def rates(ys):
    n = max(1, len(ys))
    return {"n": len(ys), "median_x": round(10 ** statistics.median(ys), 2) if ys else None,
            "p3": round(sum(v >= VIRAL for v in ys) / n, 3), "pflop": round(sum(v <= -VIRAL for v in ys) / n, 3)}


def calib(cv, y):
    """按留一预测在 CAL 边界里的位置分 8 档（正好在边界上归上一档），每档的实际历史。"""
    bins = [[] for _ in range(len(CAL) + 1)]
    for p, v in zip(cv, y):
        bins[bisect.bisect_right(CAL, p)].append(v)
    return {"edges": CAL, "bins": [rates(ys) for ys in bins]}


def fit(rows, feats):
    """rows 已带 lift；feats {id: 答案}。返回模型 dict（不含任何作者或帖子信息）。"""
    rows = [r for r in rows if r["id"] in feats]
    X = [vec(feats[r["id"]]) for r in rows]
    cols = list(zip(*X))
    mu = [statistics.fmean(c) for c in cols]
    sd = [statistics.pstdev(c) or 1.0 for c in cols]
    Z = [[(a - m) / s for a, m, s in zip(x, mu, sd)] + [1.0] for x in X]
    y, g = [r["lift"] for r in rows], [r["user"].lower() for r in rows]
    w, cv = ridge_cv(Z, y, g)
    per = [spearman([cv[i] for i in idx], [y[i] for i in idx])
           for idx in ([i for i in range(len(g)) if g[i] == a] for a in set(g)) if len(idx) >= 15]
    return {"jev": JEV, "reader": READER, "trained": time.strftime("%Y-%m-%d"), "n": len(rows), "authors": len(set(g)),
            "viral": sum(v >= VIRAL for v in y), "rho": round(spearman(cv, y), 3),
            "rho_author_median": round(statistics.median(per), 3) if per else None,
            "cols": COLS, "mu": mu, "sd": sd, "w": w, "calib": calib(cv, y)}


def bin_name(k):
    lo, hi = ([None] + CAL)[k], (CAL + [None])[k]
    return f"<{10 ** hi:.2f}" if lo is None else f"≥{10 ** lo:.2f}" if hi is None else f"{10 ** lo:.2f}–{10 ** hi:.2f}"


def show_model(m):
    print(f"{m['n']} 条、{m['authors']} 位作者、爆款（≥3 倍）{m['viral']} 条；按作者留一 ρ {m['rho']}，作者内中位 ρ {m['rho_author_median']}")
    print("校准表（预测倍数 → 历史条数 → 真过 3 倍 / 真跌到 1/3 以下）：")
    for k, b in enumerate(m["calib"]["bins"]):
        print(f"  {bin_name(k) + ' 倍':<12} n={b['n']:<6} ≥3 倍 {b['p3']:.1%}  ≤1/3 倍 {b['pflop']:.1%}")


def train(yes):
    rows, feats = labeled(load_posts()), load_feats()
    todo = [r for r in rows if r["id"] not in feats]
    if len({r["user"].lower() for r in rows}) < 3:
        raise SystemExit("合格帖不够：至少要 3 位作者、每人 ≥8 条发出满 48 小时的中文帖，先多 pull 几个号")
    if todo:
        print(f"{len(todo)} 条帖需要 Jev 特征，预计费用约 ${len(todo) * 6300 * 0.042 / 1e6:.2f}（每条约 6,300 输入 token × $0.042 / 百万）")
        if not yes and input("继续？[y/N] ").strip().lower() != "y":
            raise SystemExit("已取消")
        with FEATS.open("a") as out, ThreadPoolExecutor(8) as pool:
            for r, a in zip(todo, pool.map(safe_ask, todo)):
                if a:
                    feats[r["id"]] = a
                    out.write(json.dumps({"id": r["id"], "a": a, "m": JEV}) + "\n")
                    out.flush()
    m = fit(rows, feats)
    (DIR / "model.json").write_text(json.dumps(m, ensure_ascii=False))
    show_model(m)


# ---------- 打分 ----------
def gap_note(gap):
    return "最高" if gap <= 0 else f"低 {gap:.3f}，" + next(t for lim, t in GAPS if gap < lim)


def hist_of(m, pred):
    return m["calib"]["bins"][bisect.bisect_right(m["calib"]["edges"], pred)]


def load_model():
    f = DIR / "model.json"
    m = json.loads((f if f.exists() else BUNDLED).read_text())
    if m["cols"] != COLS:
        raise SystemExit("model.json 的问题列和当前脚本不一致，请重新 train")
    if m.get("reader") != READER:
        print("警告：模型训练时的读者描述和脚本里的 READER 不同，分数可能不准；请重新 train")
    return m


def score(paths):
    m = load_model()
    res = []
    for p in paths:
        text = Path(p).read_text().strip()
        z = [(a - u) / s for a, u, s in zip(vec(ask(text, model=m["jev"])), m["mu"], m["sd"])]
        res.append((p, z, sum(a * b for a, b in zip(m["w"], z + [1.0]))))
    best = max(r[2] for r in res)
    print(f"模型：{m['n']} 条、{m['authors']} 位作者，按作者留一 ρ {m['rho']}。只能排序、避雷，不能预言。\n")
    for p, _, pred in sorted(res, key=lambda r: -r[2]):
        b = hist_of(m, pred)
        print(f"{p}\n  预测 {10 ** pred:.2f} 倍作者近期中位 · 历史同档 {b['n']} 条里过 3 倍 {b['p3']:.0%}、≤1/3 倍 {b['pflop']:.0%}"
              f" · {gap_note(best - pred)}")
    p, z, _ = max(res, key=lambda r: r[2])
    label = lambda c: NAMES.get(COLS[c].split("=")[0], COLS[c].split("=")[0]) + (
        "=" + ZH.get(COLS[c].split("=")[1], COLS[c].split("=")[1]) if "=" in COLS[c] else "")
    contrib = sorted(((m["w"][c] * z[c], c) for c in range(len(COLS))), reverse=True)
    tag = lambda c: label(c) + ("偏多" if z[c] > 0 else "偏少")  # 跟语料平均比
    print(f"\n最高版 {p} 的贡献（权重 × 标准化答案；偏多 / 偏少是跟语料平均比）")
    print("  加分：" + "；".join(f"{tag(c)} {v:+.3f}" for v, c in contrib[:5]))
    print("  扣分：" + "；".join(f"{tag(c)} {v:+.3f}" for v, c in contrib[::-1][:5]))


# ---------- 发后查看 ----------
def early(url):
    mt = XURL.search(url)
    if not mt:
        raise SystemExit("需要 x.com/用户/status/数字 形式的链接")
    user, sid = mt.groups()
    t = get(f"https://api.fxtwitter.com/{user}/status/{sid}").get("tweet") or {}
    views, ts = t.get("views"), t.get("created_timestamp")
    if views is None or not ts:
        raise SystemExit("读不到这条帖的浏览或发帖时间")
    age = (time.time() - ts) / 3600
    if age < 12:
        raise SystemExit(f"已发 {age:.1f} 小时，太早，满 12 小时再看")
    time.sleep(0.5)
    now = time.time()
    vs = [r["views"] for r in timeline(user, 2) if r["views"] is not None and r["ts"] and now - r["ts"] >= 48 * 3600 and r["id"] != sid]
    if not vs:
        raise SystemExit("作者时间线前两页里没有发出 ≥48 小时的帖，算不出中位")
    med = statistics.median(vs)
    proj = views * (1.29 if age < 24 else 1.05 if age < 48 else 1.0)
    k = proj / max(med, 1)
    print(f"浏览 {views:,}，已发 {age:.1f} 小时；作者发出 ≥48 小时的 {len(vs)} 条帖浏览中位 {med:,.0f}")
    print(f"推算最终浏览 {proj:,.0f}，是作者中位的 {k:.2f} 倍 → {'过' if k >= 3 else '没过'} 3 倍")


# ---------- 自检 ----------
def selftest():
    assert ranks([3, 1, 2, 2]) == [3, 0, 1.5, 1.5] and abs(spearman([1, 2, 3], [5, 6, 9]) - 1) < 1e-9
    day = 86400
    mk = lambda i, u, views, d: {"id": f"{u}{i}", "user": u, "views": views, "ts": d * day, "fetched": 100 * day, "media": [],
                                 "quote": False, "article": False, "text": "这是一条用来测试的中文帖子内容第" + str(i) + "条"}
    posts = {r["id"]: r for r in [mk(i, "a", 990, i) for i in range(9)] + [mk(9, "a", 9990, 5)] + [mk(i, "b", 90, i) for i in range(5)]
             + [mk(0, "c", 5, 3)]}
    posts["a0"]["fetched"] = posts["a0"]["ts"] + 3600  # 不足 48 小时，不收
    lab = {r["id"]: r for r in labeled(posts)}
    assert "a0" not in lab and not any(k[0] == "b" for k in lab), "过新、作者不足 8 条的帖不该进"
    assert abs(lab["a9"]["lift"] - 1.0) < 1e-9 and abs(lab["a1"]["lift"] - (math.log10(1000) - math.log10(1000))) < 0.01
    import random
    random.seed(1)
    X = [[random.gauss(0, 1) for _ in range(3)] for _ in range(300)]
    y = [2 * x[0] - x[1] + random.gauss(0, 0.1) for x in X]
    w, cv = ridge_cv([x + [1.0] for x in X], y, [i % 5 for i in range(300)], lam=0.01)
    assert w[0] > 1.8 and w[1] < -0.8 and abs(w[2]) < 0.1 and spearman(cv, y) > 0.95, w
    cb = calib([-0.5, 0, 0.1, 0.449, 0.45, 2.0], [0.5, -0.5, 0, 0, 1, 0.48])
    assert [b["n"] for b in cb["bins"]] == [1, 1, 1, 0, 0, 0, 1, 2] and cb["bins"][0]["p3"] == 1.0 and cb["bins"][1]["pflop"] == 1.0
    m = {"calib": cb}
    assert hist_of(m, -1)["n"] == 1 and hist_of(m, 0.45)["n"] == 2 and hist_of(m, 9)["n"] == 2
    assert gap_note(0) == "最高" and "分不开" in gap_note(0.14) and "6 成" in gap_note(0.2) and "3/4" in gap_note(0.3)
    assert len(vec({})) == len(COLS)
    print("selftest 通过")


def main(argv):
    load_env()
    cmd, args = (argv[0] if argv else ""), argv[1:]
    opt = lambda n, d=None: args[args.index(n) + 1] if n in args else d
    need_key = lambda: "JEV_API_KEY" in os.environ or sys.exit("缺 JEV_API_KEY：去 https://console.typesafe.ai/keys 开 key")
    if cmd == "selftest":
        selftest()
    elif cmd == "pull" and opt("--authors"):
        pull(opt("--authors"), int(opt("--pages", 20)))
    elif cmd == "train":
        need_key()
        train("--yes" in args)
    elif cmd == "score" and args:
        need_key()
        score(args)
    elif cmd == "early" and args:
        early(args[0])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
