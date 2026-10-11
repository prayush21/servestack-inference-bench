#!/usr/bin/env python3
"""Generate configs/prompts.jsonl: a fixed, versioned prompt set, pre-templated for Qwen2.5.

24 prompts in three input-length buckets (short / medium / long) so that prefill cost varies
realistically while staying deterministic. All text is original to this repository.
Run from the repo root:  python scripts/build_prompts.py   (tests assert the file is up to date)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmbench.prompts import qwen_chat  # noqa: E402

SHORT = [
    "Explain what a KV cache is in one paragraph.",
    "Write a haiku about a GPU running out of memory.",
    "List five practical tips for writing clear commit messages.",
    "What is the difference between latency and throughput?",
    "Give three reasons a web service might return HTTP 503.",
    "Describe how a hash map handles collisions.",
    "Summarize the purpose of a Kubernetes readiness probe.",
    "Suggest a name for a coffee shop run by robots, and explain it.",
]

MEDIUM = [
    "A small team runs a REST API on a single virtual machine. Traffic doubles every Monday "
    "morning, response times spike, and on-call engineers restart the service by hand. They have "
    "a limited budget and no dedicated platform engineer. Propose a step-by-step plan to make the "
    "Monday spike a non-event, explaining the trade-offs of each step.",
    "You are reviewing a pull request that adds retry logic to a payment client. The retry loop "
    "retries on every exception, up to ten times, with no delay between attempts, and logs the "
    "full request body on each failure. Write a constructive code-review comment that explains "
    "the risks and suggests concrete improvements.",
    "Compare batch processing and stream processing for a retailer that wants to detect "
    "fraudulent card transactions. Cover latency requirements, operational complexity, cost, and "
    "how each approach handles late-arriving data. End with a recommendation.",
    "A student says: 'Big-O notation tells you how fast a program runs.' Explain gently why this "
    "is incomplete, using an example where an O(n log n) algorithm is slower in practice than an "
    "O(n^2) algorithm, and what Big-O does tell you.",
    "Write a short onboarding guide for a new engineer joining a team that maintains a "
    "machine-learning feature store. Include what to read first, which dashboards to watch, how "
    "to safely test changes, and who to ask for help.",
    "Explain to a product manager why adding more GPUs does not always make an LLM chatbot "
    "respond faster for a single user, and what kinds of changes would actually reduce the time "
    "the user waits for the first word of the answer.",
    "Draft an incident postmortem summary for an outage where a configuration change set a cache "
    "TTL to zero, causing the database to be overloaded for forty minutes. Use a blameless tone and "
    "include timeline, impact, root cause, and three follow-up actions.",
    "Describe how you would design a rate limiter for a public API that has free and paid tiers. "
    "Discuss token bucket versus sliding window approaches, where state should live, and how to "
    "communicate limits to clients.",
]

_LONG_CONTEXT = [
    (
        "Design notes: the ingestion service receives sensor readings from roughly twelve thousand "
        "field devices. Each device sends a batch every thirty seconds containing a device "
        "identifier, a monotonic sequence number, a timestamp from the device clock, and between "
        "one and forty readings. Device clocks drift by up to two minutes per week and are corrected "
        "only when a technician visits. The current pipeline writes each batch directly to a "
        "relational database inside a single transaction. During regional network outages, devices "
        "buffer data locally and later upload several hours of batches at once, which has caused the "
        "database to fall behind by up to forty minutes. Duplicate batches occur when a device times "
        "out waiting for an acknowledgement and resends. Downstream, an alerting job scans the last "
        "five minutes of readings every minute and pages an operator if any reading exceeds a "
        "threshold. Operators have complained about two problems: alerts that fire late after "
        "outages, and occasional duplicate pages for the same event. The team is considering "
        "introducing a durable message queue between the devices and the database, switching the "
        "alerting job to consume events from the queue, and deduplicating on the pair of device "
        "identifier and sequence number. They are unsure whether to use device timestamps or server "
        "receive timestamps for alert windows, and whether exactly-once delivery is required."
    ),
    (
        "Meeting transcript summary: the mobile team reported that the app's cold start time grew "
        "from 1.8 seconds to 3.1 seconds over the last three releases. Profiling shows that the "
        "largest contributors are synchronous initialization of an analytics SDK, eager loading of a "
        "large localization bundle, and a network call that fetches feature flags before the first "
        "screen renders. The analytics owner says the SDK must initialize early to capture install "
        "attribution. The localization owner notes that only the active language is needed at "
        "startup, but the bundle format makes partial loading awkward. The feature-flag call exists "
        "because a past incident showed an experimental screen to users who should not have seen it. "
        "The design team wants a skeleton screen shown within one second. Leadership has asked for a "
        "plan that restores cold start below two seconds within two releases without reintroducing "
        "the feature-flag incident or losing attribution data. QA has limited device coverage for "
        "older Android phones, which are the most affected. There is disagreement on whether to "
        "cache feature flags on disk from the previous session and accept slightly stale values."
    ),
    (
        "Policy excerpt: employees may work remotely up to three days per week with manager "
        "approval. Teams must define at least one shared in-office day. Equipment purchased for home "
        "offices is reimbursed up to a fixed annual amount, provided receipts are submitted within "
        "sixty days of purchase. Employees working from another country for more than fourteen "
        "consecutive days must obtain prior approval from the people team because of tax and legal "
        "obligations. Customer data may only be accessed from company-managed devices and never over "
        "public networks without the company VPN. Managers are expected to evaluate performance on "
        "outcomes rather than visible hours, and to hold a check-in at least every two weeks with each "
        "direct report. Exceptions to the in-office day may be granted for medical reasons, caregiving "
        "responsibilities, or approved accommodations, and these exceptions are confidential between "
        "the employee, their manager, and the people team. The policy will be reviewed annually, and "
        "feedback can be submitted anonymously through the internal survey tool."
    ),
    (
        "Experiment log: we fine-tuned a small text classifier to route customer support tickets "
        "into eight queues. The baseline keyword router achieved 71 percent accuracy on a held-out "
        "set of two thousand tickets labeled by senior agents. The first fine-tuned model reached 88 "
        "percent accuracy but performed poorly on the billing-dispute queue, where recall was only 54 "
        "percent. Investigation showed that billing-dispute tickets made up just three percent of the "
        "training data and were often mislabeled as general billing questions. After relabeling four "
        "hundred tickets and oversampling the minority class, overall accuracy rose to 89 percent and "
        "billing-dispute recall rose to 79 percent, while precision on general billing fell from 92 "
        "to 87 percent. Agents reported that misrouted billing disputes are expensive because they "
        "often escalate to chargebacks. The team is debating whether to deploy the new model, add a "
        "confidence threshold that sends uncertain tickets to a human triage queue, or collect more "
        "data first. Latency is not a concern, but the triage queue is staffed only during business "
        "hours in one time zone."
    ),
]

_LONG_TASKS = [
    "Summarize the situation in five bullet points, then recommend what the team should do first and why.",
    "Identify the three biggest risks described above and propose a mitigation for each.",
]

LONG = [f"{ctx}\n\n{task}" for ctx in _LONG_CONTEXT for task in _LONG_TASKS]


def build() -> list[dict[str, str]]:
    rows = []
    for category, items in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
        for i, user in enumerate(items):
            rows.append({"id": f"{category}-{i:02d}", "category": category, "user": user,
                         "prompt": qwen_chat(user)})
    # Interleave buckets so every concurrency level sees a mix even with few requests.
    by_cat = [[r for r in rows if r["category"] == c] for c in ("short", "medium", "long")]
    return [r for group in zip(*by_cat) for r in group]


def render() -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build())


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "configs" / "prompts.jsonl"
    out.write_text(render())
    print(f"wrote {out} ({len(build())} prompts)")
