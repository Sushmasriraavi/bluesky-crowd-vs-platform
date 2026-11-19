"""Platform actions from the official labeler, as per-account flags in a window after each account's index time.

The labeler does much more than takedowns (see probe D1): needs-review flags, account takedowns, temporary
suspensions, and labels on posts - adult content (porn/sexual/nudity) and abuse (rude/intolerant/threat).
Account-level reason labels are almost never used, so the type of harm is read from post labels, not from
takedown reasons.
"""
from __future__ import annotations

# name -> (description, SQL condition on the unioned label events  e(lvl, val))
ACTIONS = {
    "review":    ("flagged for review (needs-review)",       "e.lvl = 'acct' AND e.val = 'needs-review'"),
    "abuse":     ("abuse label (rude/intolerant/threat)",    "e.val IN ('rude', 'intolerant', 'threat')"),
    "adult":     ("adult-content label (porn/sexual/nudity)", "e.val IN ('porn', 'sexual', 'nudity', 'sexual-figurative')"),
    "post_removed": ("post taken down",                       "e.lvl = 'rec' AND e.val = '!takedown'"),
    "suspend":   ("temporary suspension",                    "e.lvl = 'acct' AND e.val = '!suspend'"),
    "takedown":  ("account taken down",                      "e.lvl = 'acct' AND e.val = '!takedown'"),
    "any":       ("any platform action",                     "true"),
}
ORDER = ["any", "review", "adult", "abuse", "post_removed", "suspend", "takedown"]
MODEL_OUTCOMES = ["takedown", "review", "abuse", "adult"]   # common enough to train a model on


def events_sql(ae: str, re_: str) -> str:
    """Account-level and record-level label events in one relation (non-negated only)."""
    return f"""(SELECT did, val, ts, 'acct' AS lvl FROM {ae} WHERE NOT neg
               UNION ALL
               SELECT did, val, ts, 'rec' AS lvl FROM {re_} WHERE NOT neg)"""


def flags_sql(cohort_sql: str, ae: str, re_: str, until: str, actions=None) -> str:
    """SELECT did, act_<name> (bool) for each action applied in (index_t, <until>].

    cohort_sql must yield did, index_t. `until` is an SQL expression that may use c.index_t,
    e.g. "c.index_t + INTERVAL 30 DAY".
    """
    actions = actions or ORDER
    aggs = ",\n".join(f"coalesce(bool_or(e.did IS NOT NULL AND ({ACTIONS[a][1]})), false) AS act_{a}" for a in actions)
    return f"""
        WITH c AS (SELECT did, index_t FROM ({cohort_sql})),
             e AS (SELECT e.did, e.val, e.lvl FROM {events_sql(ae, re_)} e
                   JOIN c ON c.did = e.did AND e.ts > c.index_t AND e.ts <= {until})
        SELECT c.did, {aggs}
        FROM c LEFT JOIN e ON e.did = c.did GROUP BY c.did"""
