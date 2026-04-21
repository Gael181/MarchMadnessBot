"""
Prompt template system for intent-based response generation.

This module defines prompt templates for different types of questions:
- Factual Lookup: General Q&A about teams/seasons
- Team Comparison: Head-to-head team analysis
- Seed Matchup: Specific seed-based matchup analysis (e.g., 12 vs 5)
- Trend Analysis: Historical upset patterns and trends
- Team Stats: Focused team statistics and records
- Prediction: Hypothetical matchup predictions
- Rules Explanation: Tournament structure and rule explanations
"""

from enum import Enum
from typing import Optional, Dict, Any


class QuestionIntent(Enum):
    """Enum for all supported question intent types."""
    FACTUAL_LOOKUP = "factual_lookup"
    TEAM_COMPARISON = "team_comparison"
    SEED_MATCHUP = "seed_matchup"
    TREND_ANALYSIS = "trend_analysis"
    TEAM_STATS = "team_stats"
    PREDICTION = "prediction"
    RULES_EXPLANATION = "rules_explanation"


class PromptTemplate:
    """
    Base class for prompt templates.
    
    Subclasses define specific prompt structures, default parameters,
    and validation logic for different question types.
    """
    
    # Default template; subclasses should override
    template: str = ""
    
    # Default temperature for LLM generation; can be overridden per instance
    default_temperature: float = 0.7
    
    # Parameters this prompt template supports (e.g., "style", "constraints")
    parameters: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        """
        Initialize prompt template with optional parameter overrides.
        
        Args:
            **kwargs: Optional parameter overrides (e.g., temperature=0.5)
        """
        self.parameters = {**self.parameters}
        if "temperature" in kwargs:
            self.default_temperature = kwargs["temperature"]
    
    def build(self, **kwargs) -> str:
        """
        Build the final prompt string by interpolating context and parameters.
        
        Args:
            **kwargs: Variables to interpolate into the template
                (e.g., question, context, team1, team2, etc.)
        
        Returns:
            Final prompt string ready for LLM consumption
        """
        return self.template.format(**kwargs)
    
    def get_temperature(self) -> float:
        """Return the temperature for this prompt type."""
        return self.default_temperature


class FactualLookupPrompt(PromptTemplate):
    """
    General factual Q&A about teams, seasons, and tournament data.
    
    Migrated from llm_service.generate_grounded_answer().
    Used for: generic questions, team lookups, season stats, etc.
    """
    
    default_temperature = 1.0
    
    template = """You are a college basketball assistant.

Rules:
1. Use the dataset context as your primary evidence.
2. Explicitly reference supporting evidence from the dataset when available.
3. Do not invent statistics or seasons not present in the dataset context.
4. If the dataset context is insufficient, say so clearly.
5. If the question is general and outside the dataset, you may answer generally, but say that the answer is not grounded in the dataset.

User question:
{question}

Dataset context:
{context}
"""


class TeamComparisonPrompt(PromptTemplate):
    """
    Head-to-head comparison of two teams.
    
    Migrated from llm_service.generate_comparison_answer().
    Used for: "compare Team A vs Team B" queries.
    """
    
    default_temperature = 1.0
    
    template = """You are a college basketball analyst.

Compare the following two teams using ONLY the dataset context.

Rules:
1. Use only the provided context.
2. Do not invent stats.
3. Clearly compare strengths and weaknesses.
4. Mention specific stats when possible.
5. End with a short conclusion.

Team 1: {team1}
Context:
{context1}

Team 2: {team2}
Context:
{context2}

Provide:
- Key statistical comparison
- Tournament-related insights if available
- Final comparison summary
"""


class SeedMatchupPrompt(PromptTemplate):
    """
    Analysis of seed-based matchups (e.g., 12 vs 5 seeds).
    
    Specialized prompt for: "#X seed vs #Y seed" questions.
    Focuses on historical outcomes and matchup dynamics.
    """
    
    default_temperature = 0.7
    
    template = """You are a college basketball analytics expert specializing in seed-based matchups.

Analyze the following seed matchup using ONLY dataset context.

Rules:
1. Use ONLY provided dataset context.
2. Focus on historical outcomes between these seed numbers.
3. Identify patterns: how often does the higher seed win? Upset frequency?
4. Provide specific examples from the dataset when available.
5. Do NOT invent statistics.
6. If insufficient data, say so clearly.

Seed Matchup: {seed_a} seed vs {seed_b} seed

Dataset Context:
{context}

Provide:
- Historical win rate for higher vs lower seed
- Upset frequency for this matchup
- Notable patterns or trends
- Confidence in predictions based on dataset size
"""


class TrendAnalysisPrompt(PromptTemplate):
    """
    Historical upset patterns and tournament trends.
    
    Migrated/adapted from llm_service.generate_trend_answer().
    Used for: "How often do 12 seeds beat 5 seeds?" type questions.
    """
    
    default_temperature = 0.7
    
    template = """You are a college basketball analytics expert.

You are analyzing HISTORICAL MARCH MADNESS TRENDS.

Rules:
1. ONLY use dataset context.
2. Focus on patterns across multiple games/seasons.
3. Identify upset frequency patterns (e.g., 12 vs 5, 11 vs 6).
4. Summarize trends clearly and concisely.
5. If data is insufficient, explicitly say so.
6. Do NOT invent statistics.

User question:
{question}

Dataset Context:
{context}

Provide:
- Most common upset types
- Frequency patterns observed in the dataset
- Notable anomalies
- Brief predictive insight for brackets
"""


class TeamStatsPrompt(PromptTemplate):
    """
    Focused statistics and records for a specific team.
    
    Used for: "What are [Team]'s stats?" or "Tell me about [Team]" queries.
    Emphasizes filtering and formatting available statistics.
    """
    
    default_temperature = 0.5
    
    template = """You are a college basketball statistician.

Provide detailed statistics and records for the requested team using ONLY the dataset context.

Rules:
1. Use ONLY dataset context provided.
2. Organize stats clearly (e.g., season record, conference, seed, tournament results).
3. Do not invent or estimate statistics.
4. If specific stats are not in the context, omit them rather than guess.
5. Highlight tournament appearances and performance if available.

Team: {team}

Dataset context:
{context}

Provide:
- Season records and performance metrics
- Conference and tournament information
- Any notable achievements or trends
- Caveats about data completeness
"""


class PredictionPrompt(PromptTemplate):
    """
    Hypothetical matchup predictions based on historical data.
    
    Used for: "How would [Team A] fare against [Team B]?" type questions.
    Emphasizes historical patterns while clearly caveating speculative nature.
    """
    
    default_temperature = 0.8
    
    template = """You are a college basketball analyst providing data-informed predictions.

Predict the outcome of a hypothetical matchup using historical patterns from the dataset.

IMPORTANT: This is speculative analysis based on historical data, not a guaranteed outcome.

Rules:
1. Use ONLY dataset context as evidence.
2. Base predictions on historical performance trends.
3. Clearly state confidence levels and caveats.
4. Do not invent statistics.
5. Acknowledge factors NOT captured in the dataset (e.g., injuries, current form).
6. End with a clear disclaimer about prediction uncertainty.

Hypothetical Matchup:
{team1} vs {team2}

Dataset context:
{context}

Provide:
- Historical head-to-head record if available
- Key statistical advantages and disadvantages
- Prediction with confidence level (low/medium/high)
- Disclaimer about uncertainty and unmodeled factors
"""


class RulesExplanationPrompt(PromptTemplate):
    """
    Explanations of tournament structure, rules, and terminology.
    
    Used for: "What does a 12 seed mean?" or "How does seeding work?" queries.
    Factual, educational tone with dataset examples when relevant.
    """
    
    default_temperature = 0.5
    
    template = """You are a college basketball rules and tournament structure expert.

Explain tournament concepts, rules, and terminology. Use dataset examples when helpful.

Rules:
1. Provide clear, accurate explanations of tournament structure.
2. Use dataset examples to illustrate concepts when relevant.
3. Keep explanations concise and accessible.
4. Do not speculate beyond tournament rules; stick to facts.
5. If examples from dataset are not available, provide general explanation.

Question:
{question}

Dataset context (if relevant):
{context}

Provide:
- Clear definition or explanation
- How it applies in practice
- Examples from dataset or general March Madness
- Relevant context for brackets/predictions
"""


class PromptSelector:
    """
    Selects and instantiates the appropriate prompt template based on question intent.
    
    Maps QuestionIntent enums to prompt template classes and manages template
    instantiation with parameters.
    """
    
    # Mapping of intent to prompt class
    INTENT_TO_PROMPT = {
        QuestionIntent.FACTUAL_LOOKUP: FactualLookupPrompt,
        QuestionIntent.TEAM_COMPARISON: TeamComparisonPrompt,
        QuestionIntent.SEED_MATCHUP: SeedMatchupPrompt,
        QuestionIntent.TREND_ANALYSIS: TrendAnalysisPrompt,
        QuestionIntent.TEAM_STATS: TeamStatsPrompt,
        QuestionIntent.PREDICTION: PredictionPrompt,
        QuestionIntent.RULES_EXPLANATION: RulesExplanationPrompt,
    }
    
    @staticmethod
    def get_prompt(intent: QuestionIntent, **kwargs) -> PromptTemplate:
        """
        Get a prompt template instance for the given intent.
        
        Args:
            intent: QuestionIntent enum value
            **kwargs: Optional parameter overrides (e.g., temperature=0.5)
        
        Returns:
            PromptTemplate instance ready to use
            
        Raises:
            ValueError: If intent is not recognized
        """
        if intent not in PromptSelector.INTENT_TO_PROMPT:
            raise ValueError(f"Unknown question intent: {intent}")
        
        prompt_class = PromptSelector.INTENT_TO_PROMPT[intent]
        return prompt_class(**kwargs)
    
    @staticmethod
    def get_all_intents():
        """Return list of all supported question intents."""
        return list(PromptSelector.INTENT_TO_PROMPT.keys())
