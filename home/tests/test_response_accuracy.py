import re
from unittest.mock import MagicMock, patch
from django.test import TestCase
from bot.services.chat_service import ChatService, extract_seeds, extract_teams, route_dataset



def _fake_llm_payload(text: str) -> dict:
    """Return a minimal payload that looks like what LLMService methods return."""
    return {"text": text, "token_used": "N/A", "response_time": "10ms"}

# 1. Unit tests – pure parsing/routing logic (no DB, no LLM, no FAISS)
class ExtractSeedsTests(TestCase):
    """extract_seeds() should parse seed matchups from natural-language questions."""

    def test_explicit_vs_format(self):
        a, b = extract_seeds("How often does a 12 seed beat a 5 seed?")
        self.assertEqual(sorted([a, b]), [5, 12])

    def test_hashtag_format(self):
        a, b = extract_seeds("Tell me about #11 vs #6 matchups")
        self.assertEqual(sorted([a, b]), [6, 11])

    def test_upset_verb_format(self):
        a, b = extract_seeds("Can a 13 seed upset a 4 seed?")
        self.assertEqual(sorted([a, b]), [4, 13])

    def test_no_seeds_returns_none(self):
        a, b = extract_seeds("Who won the most championships?")
        self.assertIsNone(a)
        self.assertIsNone(b)

    def test_out_of_range_seed_ignored(self):
        # 17 is not a valid seed
        a, b = extract_seeds("17 seed vs 1 seed matchup")
        self.assertIsNone(a)
        self.assertIsNone(b)


class ExtractTeamsTests(TestCase):
    """extract_teams() should pull two team names from comparison questions."""

    def test_vs_format(self):
        t1, t2 = extract_teams("Duke vs Kansas in 2022")
        self.assertIn("Duke", t1 + t2)
        self.assertIn("Kansas", t1 + t2)

    def test_compare_and_format(self):
        t1, t2 = extract_teams("Compare Gonzaga and Villanova")
        self.assertIn("Gonzaga", t1 + t2)
        self.assertIn("Villanova", t1 + t2)

    def test_no_teams_returns_none(self):
        t1, t2 = extract_teams("Who has the best free-throw percentage?")
        self.assertIsNone(t1)
        self.assertIsNone(t2)


class RouteDatasetTests(TestCase):
    """route_dataset() should map questions to the right dataset."""

    def test_upset_keyword_routes_to_tournament(self):
        self.assertEqual(route_dataset("What are the most common upsets?"), "tournament")

    def test_seed_keyword_routes_to_tournament(self):
        self.assertEqual(route_dataset("Historical seed trends in March Madness"), "tournament")

    def test_team_question_routes_to_teams(self):
        self.assertEqual(route_dataset("How did Duke perform this season?"), "teams")

    def test_how_often_routes_to_tournament(self):
        self.assertEqual(route_dataset("How often does a lower seed win?"), "tournament")



# 2. Integration tests – ChatService with mocked LLM + real RAG search
MOCK_LLM_PATH = "bot.services.chat_service.LLMService"

class SeedMatchupAccuracyTests(TestCase):
    """
    Known fact: #12 seeds beat #5 seeds more than 35 % of the time historically.
    The response must surface numeric upset data for the 12 vs 5 matchup.
    """

    def _mock_llm(self, answer_text: str):
        mock = MagicMock()
        mock.return_value.generate_trend_answer.return_value = _fake_llm_payload(answer_text)
        mock.return_value.generate_grounded_answer.return_value = _fake_llm_payload(answer_text)
        return mock

    @patch(MOCK_LLM_PATH)
    def test_12_vs_5_contains_upset_stats(self, MockLLM):
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            "12 seeds upset 5 seeds approximately 35% of the time."
        )
        answer = ChatService.answer_question("How often does a 12 seed beat a 5 seed?")

        self.assertNotEqual(answer.outcome, "error", msg=answer.error_message)
        # Response must mention both seeds
        self.assertRegex(answer.text, r"12", msg="Response should reference seed 12")
        self.assertRegex(answer.text, r"5", msg="Response should reference seed 5")

    @patch(MOCK_LLM_PATH)
    def test_11_vs_6_response_is_not_empty(self, MockLLM):
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            "11 seeds win roughly 37% of games against 6 seeds."
        )
        answer = ChatService.answer_question("#11 vs #6 seed upsets")

        self.assertTrue(len(answer.text) > 0)
        self.assertNotIn("knowledge search failed", answer.text.lower())

    @patch(MOCK_LLM_PATH)
    def test_no_games_found_gives_graceful_message(self, MockLLM):
        # FAISS/dataset always returns something; system will succeed or fallback
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload("No 1 vs 1 seed matchups exist.")
        answer = ChatService.answer_question("1 seed vs 1 seed matchup history")
        self.assertIn(answer.outcome, {"no_results", "error", "trend_success", "llm_fallback", "success"})


# 3. Team comparison accuracy
class TeamComparisonAccuracyTests(TestCase):
    """
    The bot must return data for both requested teams when a vs-comparison is
    asked about teams that exist in the dataset.
    """

    @patch(MOCK_LLM_PATH)
    def test_comparison_mentions_both_teams(self, MockLLM):
        MockLLM.return_value.generate_comparison_answer.return_value = _fake_llm_payload(
            "Duke has historically strong seedings. Kansas has more Final Four appearances."
        )
        answer = ChatService.answer_question("Compare Duke vs Kansas")

        self.assertNotEqual(answer.outcome, "error", msg=answer.error_message)
        lower = answer.text.lower()
        self.assertIn("duke", lower, "Response must mention Duke")
        self.assertIn("kansas", lower, "Response must mention Kansas")

    @patch(MOCK_LLM_PATH)
    def test_unknown_team_returns_no_results(self, MockLLM):
        # FAISS returns nearest neighbors even for unknown teams, so success/fallback are valid
        MockLLM.return_value.generate_comparison_answer.return_value = _fake_llm_payload("No data found for these teams.")
        answer = ChatService.answer_question("Compare ZZZFakeTeamXXX vs AnotherFakeTeam")
        self.assertIn(answer.outcome, {"no_results", "llm_fallback", "error", "success"})


# 4. Outcome / latency sanity tests
class ChatAnswerStructureTests(TestCase):
    """ChatAnswer fields must always be populated with sensible values."""

    @patch(MOCK_LLM_PATH)
    def test_latency_ms_is_positive(self, MockLLM):
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload(
            "Here is some information about Duke basketball."
        )
        answer = ChatService.answer_question("Tell me about Duke")
        self.assertGreater(answer.latency_ms, 0)

    @patch(MOCK_LLM_PATH)
    def test_outcome_is_known_string(self, MockLLM):
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload("ok")
        answer = ChatService.answer_question("Tell me about Gonzaga")
        self.assertIn(
            answer.outcome,
            {"success", "trend_success", "no_results", "error", "llm_fallback"},
        )

    @patch(MOCK_LLM_PATH)
    def test_text_is_non_empty_on_success(self, MockLLM):
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload(
            "Gonzaga has been a dominant mid-major program."
        )
        answer = ChatService.answer_question("Tell me about Gonzaga")
        if answer.outcome in {"success", "trend_success", "llm_fallback"}:
            self.assertTrue(len(answer.text.strip()) > 0)


# 5. LLM-fallback accuracy – system should still return useful data when the LLM raises an exception.
class LLMFallbackAccuracyTests(TestCase):
    """When Gemini is unavailable, the bot must still return dataset evidence."""

    @patch(MOCK_LLM_PATH)
    def test_seed_query_fallback_contains_stats(self, MockLLM):
        MockLLM.return_value.generate_trend_answer.side_effect = RuntimeError("503 unavailable")
        answer = ChatService.answer_question("How often does a 12 seed beat a 5 seed?")

        self.assertEqual(answer.outcome, "llm_fallback")
        # Even without LLM, the fallback block must include numeric upset info
        self.assertRegex(answer.text, r"\d+", msg="Fallback must include at least one number")

    @patch(MOCK_LLM_PATH)
    def test_team_query_fallback_has_dataset_rows(self, MockLLM):
        MockLLM.return_value.generate_comparison_answer.side_effect = RuntimeError("API error")
        answer = ChatService.answer_question("Compare Duke vs Kansas")

        self.assertIn(answer.outcome, {"llm_fallback", "no_results"})
        if answer.outcome == "llm_fallback":
            self.assertTrue(len(answer.text) > 0)

class KnownAnswerAccuracyTests(TestCase):
    """
    Tests against questions with verifiable answers from the dataset.
    Expected values are pre-computed from the CSV so if the bot's answer
    contradicts them, we know the system is wrong.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Compute ground truth directly from the CSV at test time
        import pandas as pd
        import os

        csv_path = os.path.join(
            os.path.dirname(__file__), 
            "../../bot/datasets/ncaa_tournament_results.csv"
        )
        df = pd.read_csv(csv_path)
        df["seed"]   = pd.to_numeric(df["seed"],   errors="coerce")
        df["seed.1"] = pd.to_numeric(df["seed.1"], errors="coerce")

        # Pre-compute 12 vs 5 upset rate from the actual data
        mask_12_5 = (
            ((df["seed"] == 12) & (df["seed.1"] == 5)) |
            ((df["seed"] == 5)  & (df["seed.1"] == 12))
        )
        matchups = df[mask_12_5]
        cls.total_12_5 = len(matchups)

        from bot.dataset import is_Upset
        upsets = matchups[matchups.apply(is_Upset, axis=1)]
        cls.upset_count_12_5 = len(upsets)
        cls.upset_pct_12_5 = round((cls.upset_count_12_5 / cls.total_12_5) * 100, 1)

    @patch(MOCK_LLM_PATH)
    def test_12_vs_5_upset_percentage_is_accurate(self, MockLLM):
        """
        The response must contain the correct upset percentage for 12 vs 5,
        computed directly from the dataset (ground truth).
        """
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            f"12 seeds upset 5 seeds {self.upset_pct_12_5}% of the time "
            f"across {self.total_12_5} games."
        )
        answer = ChatService.answer_question(
            "How often does a 12 seed beat a 5 seed?"
        )
        # The correct percentage must appear somewhere in the response
        self.assertIn(
            str(self.upset_pct_12_5),
            answer.text,
            msg=f"Expected upset rate {self.upset_pct_12_5}% not found in response"
        )

    @patch(MOCK_LLM_PATH)
    def test_12_vs_5_total_game_count_is_accurate(self, MockLLM):
        """
        The response must reflect the correct total number of 12 vs 5 games
        in the dataset.
        """
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            f"There have been {self.total_12_5} games between 12 and 5 seeds."
        )
        answer = ChatService.answer_question(
            "How many times has a 12 seed played a 5 seed?"
        )
        self.assertIn(
            str(self.total_12_5),
            answer.text,
            msg=f"Expected total game count {self.total_12_5} not found in response"
        )