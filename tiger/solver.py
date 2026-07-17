"""Solver: structure-safe repair patch application (roadmap 1.5, finding F10).

Phase 1 scope: the patching PRIMITIVE only. Candidate enumeration, cost-minimal
patch selection (2.7) and repair planning arrive with Phase 2.

Guarantees over the legacy apply_repairs.py:

  - patches modify the attributes DICT, never free-text via word replacement;
  - every patch value is validated against Omega_j (and the whole patched
    record against the constraint set C) BEFORE anything is applied;
    out-of-domain or constraint-violating patches are refused and returned as
    an escalation, leaving the row untouched;
  - the title is regenerated with brand-safe colour-word replacement
    ("Red Wing Supply Red Shirt" keeps its brand) instead of blind regex
    substitution over the whole string;
  - canonical_text is rebuilt from the patched structures, never edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tiger import text_views
from tiger.schema import Schema


@dataclass
class PatchResult:
    applied: bool
    title: str
    attrs: dict
    canonical_text: str
    changed_fields: list = field(default_factory=list)
    refusal_reason: str = ""
    escalate: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def apply_attr_patch(title: str, category: str, attrs: dict, patch: dict,
                     schema: Schema) -> PatchResult:
    """Apply {field: new_value} to the attributes dict, structure-safely.

    Refuses (escalate=True, row untouched) when any value is out of domain or
    the patched record violates the constraint set C.
    """
    attrs = dict(attrs)
    title = str(title or "")

    if not patch:
        return PatchResult(False, title, attrs,
                           text_views.canonical_text(title, category, attrs),
                           refusal_reason="empty_patch")

    # ---- validate BEFORE mutating anything (Omega_j gate) ----
    normalized: dict[str, str] = {}
    for fld, value in patch.items():
        if fld not in schema.attributes:
            return PatchResult(False, title, attrs,
                               text_views.canonical_text(title, category, attrs),
                               refusal_reason=f"unknown_field:{fld}", escalate=True)
        if not schema.in_domain(fld, value):
            return PatchResult(False, title, attrs,
                               text_views.canonical_text(title, category, attrs),
                               refusal_reason=f"out_of_domain:{fld}={value!r}", escalate=True)
        normalized[fld] = schema.normalize(fld, value)

    candidate = dict(attrs)
    candidate.update(normalized)
    violations = schema.validate_attrs(category, candidate)
    if violations:
        return PatchResult(False, title, attrs,
                           text_views.canonical_text(title, category, attrs),
                           refusal_reason="constraint_violation:" +
                                          "; ".join(v.message for v in violations),
                           escalate=True)

    # ---- apply ----
    changed = []
    brands = {str(attrs.get("brand", ""))}
    new_title = title
    for fld, new_value in normalized.items():
        old_value = schema.normalize(fld, attrs.get(fld, "")) if attrs.get(fld) else ""
        if old_value == new_value:
            continue
        if fld == "color":
            # regenerate the title's colour mention rather than word-replacing
            # arbitrary text: replace only a colour word occurring OUTSIDE
            # brand names; if none exists, leave the title alone.
            present = text_views.find_color_word(new_title, schema.domain("color"), brands)
            if present:
                new_title = text_views.replace_color_word_safe(new_title, present, new_value, brands)
        changed.append(fld)

    attrs.update(normalized)
    return PatchResult(True, new_title, attrs,
                       text_views.canonical_text(new_title, category, attrs),
                       changed_fields=changed)
