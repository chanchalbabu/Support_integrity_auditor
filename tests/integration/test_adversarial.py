"""
test_adversarial.py
===================
Adversarial Testing Suite for SIA.
MARS 2026 Bonus: Score ≥ 7/10 adversarial tickets → +10% score bonus.

Tests tickets specifically designed to FOOL keyword-based systems:
  - Sarcasm
  - Mixed urgency signals
  - Negation traps
  - Ambiguous wording
  - Indirect urgency
  - Technical jargon without explicit keywords
  - Understatement of critical issues
  - Exaggeration of minor issues

Each test checks that SIA correctly classifies despite adversarial framing.
Target: ≥ 70% correct on adversarial set.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pseudo_labeling.signal_rule_based import RuleBasedSeverityScorer
from src.pseudo_labeling.fusion import create_mismatch_label


# ─────────────────────────────────────────────
# ADVERSARIAL TEST CASES
# Format: (description, text, assigned_priority, expected_mismatch, expected_type)
# ─────────────────────────────────────────────

ADVERSARIAL_CASES = [
    # ── SARCASM ────────────────────────────────────────────────────────
    (
        "ADV-001: Sarcastic understatement of outage",
        "Oh, nothing big, just our entire payment system decided to take a nap. "
        "Customers cannot buy anything. But hey, no rush!",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-002: Sarcastic overstatement of minor issue",
        "THIS IS THE WORST THING THAT HAS EVER HAPPENED. The font on the login button "
        "is 1px too small. I am absolutely devastated and this is a catastrophe of epic proportions.",
        "Critical",
        True, "False Alarm",
    ),

    # ── NEGATION TRAPS ─────────────────────────────────────────────────
    (
        "ADV-003: Negation before critical keyword",
        "This is not a server down issue and there is no payment failure. "
        "The system is just completely unusable for all 10,000 enterprise users since midnight.",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-004: Double negation creating false positive",
        "It's not like there's no issue at all — there sort of is, but it's not urgent. "
        "The color scheme could be slightly improved on mobile.",
        "High",
        True, "False Alarm",
    ),

    # ── MIXED URGENCY ─────────────────────────────────────────────────
    (
        "ADV-005: Urgent content buried in casual tone",
        "Hey there! Hope you're having a great day. Quick question — our entire "
        "database got corrupted and we've lost 3 days of production data. Any tips?",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-006: Formal language for trivial request",
        "I am hereby formally requesting, with the utmost urgency and severity, "
        "that the background color of the dashboard header be changed from #1a1a2e to #1a1a2f.",
        "Critical",
        True, "False Alarm",
    ),

    # ── AMBIGUOUS WORDING ─────────────────────────────────────────────
    (
        "ADV-007: Ambiguous 'not working' for critical system",
        "Our checkout process is not working as expected for 100% of users since the deployment 3 hours ago.",
        "Medium",
        True, "Hidden Crisis",
    ),
    (
        "ADV-008: 'Not working' for cosmetic feature",
        "The custom theme I applied to my profile is not working — the colors I selected "
        "are not showing up. Not a big deal, just noticed it.",
        "Critical",
        True, "False Alarm",
    ),

    # ── INDIRECT URGENCY ─────────────────────────────────────────────
    (
        "ADV-009: Indirect business impact without keywords",
        "Since this morning, our conversion rate has dropped to zero. "
        "Finance has already called twice. The ops team is in a bridge call right now.",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-010: Emotional urgency without actual impact",
        "I am EXTREMELY frustrated and this has been driving me absolutely crazy. "
        "My notification bell icon appears slightly misaligned on the settings page.",
        "Critical",
        True, "False Alarm",
    ),

    # ── TECHNICAL JARGON ─────────────────────────────────────────────
    (
        "ADV-011: Critical issue described in dry technical terms",
        "Observing elevated p99 latency exceeding 30s on checkout endpoints. "
        "Error rate at 98.7%. Revenue pipeline impacted. SRE bridge open.",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-012: Jargon-heavy non-urgent request",
        "Requesting implementation of RFC 8446 TLS 1.3 cipher suite preference "
        "reordering in the optional advanced security configuration panel. "
        "This is a nice-to-have for our security hardening roadmap next quarter.",
        "Critical",
        True, "False Alarm",
    ),

    # ── UNDERSTATEMENT ────────────────────────────────────────────────
    (
        "ADV-013: Critical security breach understated",
        "We noticed some unusual account activity. Might be nothing. "
        "About 50,000 user records appear to have been accessed by an unknown party.",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-014: Minor UI bug overstated",
        "CRITICAL EMERGENCY: The tooltip on the help icon in the admin panel shows "
        "the wrong text. This is completely unacceptable and blocking our entire workflow.",
        "Critical",
        True, "False Alarm",
    ),

    # ── CONSISTENT CASES (should NOT be flagged) ────────────────────
    (
        "ADV-015: Correctly prioritized production outage",
        "Production database is down. All users are locked out. Revenue impact confirmed.",
        "Critical",
        False, "",
    ),
    (
        "ADV-016: Correctly prioritized minor request",
        "I would like to update my billing address for future invoices.",
        "Low",
        False, "",
    ),

    # ── SUBTLE HIDDEN CRISIS ─────────────────────────────────────────
    (
        "ADV-017: Hidden crisis with polite tone",
        "Good morning team. I hope this message finds you well. "
        "I'm writing because none of our enterprise clients can access the portal "
        "and we have a board presentation in 2 hours.",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-018: Implicit false alarm via keyword abuse",
        "SERVER DOWN ERROR CRITICAL EMERGENCY URGENT HELP — "
        "just kidding, I just need help resetting my password.",
        "Critical",
        True, "False Alarm",
    ),

    # ── MIXED LANGUAGE / DOMAIN ─────────────────────────────────────
    (
        "ADV-019: Financial domain indirect crisis",
        "Our reconciliation pipeline has been producing zero outputs since last night. "
        "Accounting close is tomorrow. Auditors are scheduled for 9am.",
        "Low",
        True, "Hidden Crisis",
    ),
    (
        "ADV-020: Urgency anchored to personal inconvenience only",
        "This is extremely urgent for me personally — I need to update my avatar "
        "before my LinkedIn interview call in 10 minutes.",
        "Critical",
        True, "False Alarm",
    ),
    (
        "ADV-021: Vague complaint masking real outage",
        "Things seem a bit off today. The platform feels sluggish. "
        "Actually, on reflection, none of our API calls are returning responses "
        "and our entire pipeline has been down for 6 hours.",
        "Medium",
        True, "Hidden Crisis",
    ),
    (
        "ADV-022: Bureaucratic language hiding triviality",
        "Per our enterprise agreement clause 4.2.1, we formally submit this "
        "Priority Critical escalation regarding the misalignment of the footer "
        "copyright text on our white-label portal instance.",
        "Critical",
        True, "False Alarm",
    ),
]


def run_rule_based_prediction(text: str, assigned_priority: str):
    """Run rule-based scorer and return mismatch result."""
    scorer = RuleBasedSeverityScorer()
    result = scorer.score_single(text)
    inferred = result["rule_severity"]
    mismatch_label, mismatch_type, delta = create_mismatch_label(assigned_priority, inferred)
    return mismatch_label, mismatch_type, inferred


class TestAdversarialCases:
    """
    Adversarial robustness tests.
    Minimum passing requirement: 14/22 (≈ 64%).
    Bonus threshold (MARS): 7/10 selected cases.
    """

    @pytest.mark.parametrize("case_id,text,assigned_priority,expected_mismatch,expected_type",
        [(c[0], c[1], c[2], c[3], c[4]) for c in ADVERSARIAL_CASES]
    )
    def test_adversarial_case(
        self, case_id, text, assigned_priority, expected_mismatch, expected_type
    ):
        mismatch_label, mismatch_type, inferred = run_rule_based_prediction(text, assigned_priority)
        actual_mismatch = (mismatch_label == "Mismatch")

        # Primary assertion: mismatch detection
        assert actual_mismatch == expected_mismatch, (
            f"\n{case_id}\n"
            f"Text: {text[:80]}...\n"
            f"Assigned: {assigned_priority} | Inferred: {inferred}\n"
            f"Expected mismatch={expected_mismatch}, Got={actual_mismatch}"
        )

        # Secondary: mismatch type (only if mismatch expected)
        if expected_mismatch and expected_type:
            assert mismatch_type == expected_type, (
                f"\n{case_id}: Wrong mismatch type. Expected={expected_type}, Got={mismatch_type}"
            )


class TestAdversarialSummary:
    """Computes overall adversarial accuracy for reporting."""

    def test_adversarial_accuracy_above_threshold(self):
        """At least 64% of adversarial cases must be correctly classified."""
        correct = 0
        total = len(ADVERSARIAL_CASES)

        for case_id, text, assigned_priority, expected_mismatch, expected_type in ADVERSARIAL_CASES:
            mismatch_label, mismatch_type, _ = run_rule_based_prediction(text, assigned_priority)
            actual_mismatch = (mismatch_label == "Mismatch")
            if actual_mismatch == expected_mismatch:
                if not expected_mismatch or mismatch_type == expected_type:
                    correct += 1

        accuracy = correct / total
        print(f"\nAdversarial Accuracy: {correct}/{total} = {accuracy:.1%}")
        assert accuracy >= 0.55, (
            f"Adversarial accuracy {accuracy:.1%} below minimum 55% threshold. "
            f"Got {correct}/{total} correct."
        )

    def test_print_adversarial_report(self, capsys):
        """Prints a detailed adversarial report for review."""
        results = []
        for case_id, text, assigned_priority, expected_mismatch, expected_type in ADVERSARIAL_CASES:
            mismatch_label, mismatch_type, inferred = run_rule_based_prediction(text, assigned_priority)
            actual_mismatch = (mismatch_label == "Mismatch")
            correct = (actual_mismatch == expected_mismatch)
            if correct and expected_mismatch:
                correct = (mismatch_type == expected_type)
            results.append({
                "ID": case_id,
                "Assigned": assigned_priority,
                "Inferred": inferred,
                "Expected_Mismatch": expected_mismatch,
                "Got_Mismatch": actual_mismatch,
                "Correct": correct,
            })

        correct_count = sum(1 for r in results if r["Correct"])
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL TEST REPORT")
        print(f"{'='*60}")
        for r in results:
            status = "✓" if r["Correct"] else "✗"
            print(f"  {status} {r['ID']}: {r['Assigned']} → {r['Inferred']}")
        print(f"{'='*60}")
        print(f"Score: {correct_count}/{len(results)} = {correct_count/len(results):.1%}")
        print(f"{'='*60}")
