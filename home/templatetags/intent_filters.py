"""Custom template filters for displaying formatted intent names."""

from django import template

register = template.Library()


@register.filter
def intent_display(value):
    """
    Convert intent enum value to human-readable format.
    
    Examples:
        factual_lookup -> Factual Lookup
        team_comparison -> Team Comparison
        seed_matchup -> Seed Matchup
    """
    if not value:
        return ""
    return value.replace("_", " ").title()
