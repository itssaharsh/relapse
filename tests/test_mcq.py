from relapse.mcq import format_mcq


def test_format_mcq_matches_lm_eval_layout():
    p = format_mcq("biology", "  What is X? ", ["a", "b", "c", "d"])
    assert p.startswith("The following are multiple choice questions (with answers) about biology.\n\nWhat is X?\n")
    assert "A. a\nB. b\nC. c\nD. d\n" in p
    assert p.endswith("Answer:")
