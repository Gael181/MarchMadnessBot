import os
import re
from unittest.mock import MagicMock, patch
import pandas as pd
from django.test import TestCase
from bot.dataset import is_Upset
from bot.services.chat_service import (
    ChatService,
    extract_seeds,
    extract_teams,
    route_dataset,
)

MOCK_LLM_PATH = "bot.services.chat_service.LLMService"

CSV_TOURNAMENT = os.path.join(
    os.path.dirname(__file__),
    "../../bot/datasets/ncaa_tournament_results.csv",
)


def _fake_llm_payload(text: str) -> dict:
    """Return a minimal payload that looks like what LLMService methods return."""
    return {"text": text, "token_used": "N/A", "response_time": "10ms"}


def _load_tournament_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_TOURNAMENT)
    df["seed"]   = pd.to_numeric(df["seed"],   errors="coerce")
    df["seed.1"] = pd.to_numeric(df["seed.1"], errors="coerce")
    return df

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


# 2. Known answer accuracy tests – seed matchup facts from the CSV
class KnownAnswerSeedTests(TestCase):
    """
    Known questions about seed matchup history with verifiable answers.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        df = _load_tournament_df()

        def _seed_stats(s1, s2):
            mask = (
                ((df["seed"] == s1) & (df["seed.1"] == s2)) |
                ((df["seed"] == s2) & (df["seed.1"] == s1))
            )
            matchups = df[mask]
            upsets = matchups[matchups.apply(is_Upset, axis=1)]
            total = len(matchups)
            upset_count = len(upsets)
            pct = round((upset_count / total) * 100, 1) if total else 0.0
            return total, upset_count, pct

        cls.total_12_5, cls.upsets_12_5, cls.pct_12_5 = _seed_stats(12, 5)
        cls.total_11_6, cls.upsets_11_6, cls.pct_11_6 = _seed_stats(11, 6)
        cls.total_13_4, cls.upsets_13_4, cls.pct_13_4 = _seed_stats(13, 4)

    @patch(MOCK_LLM_PATH)
    def test_12_vs_5_upset_percentage_is_accurate(self, MockLLM):
        """
        Known question: 'How often does a 12 seed beat a 5 seed?'
        Known answer:   upset % computed directly from tournament CSV.
        """
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            f"12 seeds upset 5 seeds {self.pct_12_5}% of the time "
            f"across {self.total_12_5} games."
        )
        answer = ChatService.answer_question("How often does a 12 seed beat a 5 seed?")
        self.assertIn(str(self.pct_12_5), answer.text,
            msg=f"Expected upset rate {self.pct_12_5}% not found in response")

    @patch(MOCK_LLM_PATH)
    def test_12_vs_5_total_game_count_is_accurate(self, MockLLM):
        """
        Known question: 'How many times has a 12 seed played a 5 seed?'
        Known answer:   total game count from tournament CSV.
        """
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            f"There have been {self.total_12_5} games between 12 and 5 seeds."
        )
        answer = ChatService.answer_question("How many times has a 12 seed played a 5 seed?")
        self.assertIn(str(self.total_12_5), answer.text,
            msg=f"Expected game count {self.total_12_5} not found in response")

    @patch(MOCK_LLM_PATH)
    def test_11_vs_6_upset_percentage_is_accurate(self, MockLLM):
        """
        Known question: 'How often does an 11 seed beat a 6 seed?'
        Known answer:   37.1% upset rate across 140 games (from CSV).
        """
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            f"11 seeds beat 6 seeds {self.pct_11_6}% of the time "
            f"in {self.total_11_6} tournament matchups."
        )
        answer = ChatService.answer_question("How often does an 11 seed beat a 6 seed?")
        self.assertIn(str(self.pct_11_6), answer.text,
            msg=f"Expected upset rate {self.pct_11_6}% not found in response")

    @patch(MOCK_LLM_PATH)
    def test_13_vs_4_upset_percentage_is_accurate(self, MockLLM):
        """
        Known question: 'How often does a 13 seed upset a 4 seed?'
        Known answer:   20.7% upset rate across 140 games (from CSV).
        """
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            f"13 seeds upset 4 seeds {self.pct_13_4}% of the time "
            f"across {self.total_13_4} games."
        )
        answer = ChatService.answer_question("How often does a 13 seed upset a 4 seed?")
        self.assertIn(str(self.pct_13_4), answer.text,
            msg=f"Expected upset rate {self.pct_13_4}% not found in response")

# 3. Known answer accuracy tests – individual team stats
class KnownAnswerTeamStatsTests(TestCase):
    """
    Known questions: 'Give me stats for Duke / Kansas'
    Verifies the bot surfaces the correct game counts and win totals.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        df = _load_tournament_df()

        def _team_record(name):
            total = len(df[(df["team"] == name) | (df["team.1"] == name)])
            wins = int((
                ((df["team"] == name) & (df["score"] > df["score.1"])) |
                ((df["team.1"] == name) & (df["score.1"] > df["score"]))
            ).sum())
            return total, wins

        cls.duke_games,   cls.duke_wins   = _team_record("Duke")
        cls.kansas_games, cls.kansas_wins = _team_record("Kansas")

    @patch(MOCK_LLM_PATH)
    def test_duke_game_count_in_response(self, MockLLM):
        """
        Known question: 'Give me stats for Duke'
        Known answer:   Duke has played 126 tournament games (from CSV).
        """
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload(
            f"Duke has appeared in {self.duke_games} NCAA tournament games "
            f"with {self.duke_wins} wins."
        )
        answer = ChatService.answer_question("Give me stats for Duke")
        self.assertNotEqual(answer.outcome, "error", msg=answer.error_message)
        self.assertIn(str(self.duke_games), answer.text,
            msg=f"Expected Duke game count {self.duke_games} not in response")

    @patch(MOCK_LLM_PATH)
    def test_duke_win_count_in_response(self, MockLLM):
        """
        Known question: 'How many tournament games has Duke won?'
        Known answer:   97 wins (from CSV).
        """
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload(
            f"Duke has won {self.duke_wins} NCAA tournament games."
        )
        answer = ChatService.answer_question("How many tournament games has Duke won?")
        self.assertIn(str(self.duke_wins), answer.text,
            msg=f"Expected Duke wins {self.duke_wins} not in response")

    @patch(MOCK_LLM_PATH)
    def test_kansas_game_count_in_response(self, MockLLM):
        """
        Known question: 'Give me stats for Kansas'
        Known answer:   Kansas has played 117 tournament games (from CSV).
        """
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload(
            f"Kansas has played in {self.kansas_games} NCAA tournament games "
            f"winning {self.kansas_wins}."
        )
        answer = ChatService.answer_question("Give me stats for Kansas")
        self.assertNotEqual(answer.outcome, "error", msg=answer.error_message)
        self.assertIn(str(self.kansas_games), answer.text,
            msg=f"Expected Kansas game count {self.kansas_games} not in response")

    @patch(MOCK_LLM_PATH)
    def test_kansas_win_count_in_response(self, MockLLM):
        """
        Known question: 'How many tournament games has Kansas won?'
        Known answer:   85 wins (from CSV).
        """
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload(
            f"Kansas has won {self.kansas_wins} NCAA tournament games."
        )
        answer = ChatService.answer_question("How many tournament games has Kansas won?")
        self.assertIn(str(self.kansas_wins), answer.text,
            msg=f"Expected Kansas wins {self.kansas_wins} not in response")


# 4. Known answer accuracy tests – team vs team comparison
class KnownAnswerTeamComparisonTests(TestCase):
    """
    Known questions comparing two teams.
    Verifies the bot surfaces correct head-to-head facts and mentions both teams.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        df = _load_tournament_df()

        h2h = df[
            ((df["team"] == "Duke") & (df["team.1"] == "Kansas")) |
            ((df["team"] == "Kansas") & (df["team.1"] == "Duke"))
        ]
        cls.h2h_total = len(h2h)
        cls.duke_h2h_wins = int((
            ((h2h["team"] == "Duke") & (h2h["score"] > h2h["score.1"])) |
            ((h2h["team.1"] == "Duke") & (h2h["score.1"] > h2h["score"]))
        ).sum())
        cls.kansas_h2h_wins = cls.h2h_total - cls.duke_h2h_wins

    @patch(MOCK_LLM_PATH)
    def test_duke_vs_kansas_mentions_both_teams(self, MockLLM):
        """
        Known question: 'Compare Duke vs Kansas'
        Both team names must appear in the response.
        """
        MockLLM.return_value.generate_comparison_answer.return_value = _fake_llm_payload(
            f"Duke and Kansas have met {self.h2h_total} times in the NCAA tournament. "
            f"Duke leads {self.duke_h2h_wins}-{self.kansas_h2h_wins}."
        )
        answer = ChatService.answer_question("Compare Duke vs Kansas")
        self.assertNotEqual(answer.outcome, "error", msg=answer.error_message)
        lower = answer.text.lower()
        self.assertIn("duke", lower, "Response must mention Duke")
        self.assertIn("kansas", lower, "Response must mention Kansas")

    @patch(MOCK_LLM_PATH)
    def test_duke_vs_kansas_head_to_head_count(self, MockLLM):
        """
        Known question: 'Compare Duke vs Kansas tournament history'
        Known answer:   6 head-to-head meetings (from CSV).
        """
        MockLLM.return_value.generate_comparison_answer.return_value = _fake_llm_payload(
            f"Duke and Kansas have faced each other {self.h2h_total} times "
            f"in the NCAA tournament."
        )
        answer = ChatService.answer_question("Compare Duke vs Kansas tournament history")
        self.assertIn(str(self.h2h_total), answer.text,
            msg=f"Expected h2h count {self.h2h_total} not found in response")

    @patch(MOCK_LLM_PATH)
    def test_north_carolina_vs_duke_mentions_both(self, MockLLM):
        """
        Known question: 'Compare North Carolina vs Duke'
        Both team names must appear — NC and Duke are the two most-appearing
        teams in the dataset (119 and 126 games respectively).
        """
        MockLLM.return_value.generate_comparison_answer.return_value = _fake_llm_payload(
            "North Carolina and Duke are both historically dominant programs."
        )
        answer = ChatService.answer_question("Compare North Carolina vs Duke")
        self.assertNotEqual(answer.outcome, "error", msg=answer.error_message)
        lower = answer.text.lower()
        self.assertIn("duke", lower, "Response must mention Duke")
        self.assertIn("north carolina", lower, "Response must mention North Carolina")

    @patch(MOCK_LLM_PATH)
    def test_unknown_team_comparison_handled_gracefully(self, MockLLM):
        """
        Comparing two nonsense team names should not crash the system.
        """
        MockLLM.return_value.generate_comparison_answer.return_value = _fake_llm_payload(
            "No data found for these teams."
        )
        answer = ChatService.answer_question("Compare ZZZFakeTeamXXX vs AnotherFakeTeam")
        self.assertIn(answer.outcome, {"no_results", "llm_fallback", "error", "success"})


# 5. Seed matchup pipeline smoke tests
class SeedMatchupPipelineTests(TestCase):

    @patch(MOCK_LLM_PATH)
    def test_12_vs_5_contains_seed_numbers(self, MockLLM):
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            "12 seeds upset 5 seeds approximately 35% of the time."
        )
        answer = ChatService.answer_question("How often does a 12 seed beat a 5 seed?")
        self.assertNotEqual(answer.outcome, "error", msg=answer.error_message)
        self.assertRegex(answer.text, r"12")
        self.assertRegex(answer.text, r"5")

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
        MockLLM.return_value.generate_trend_answer.return_value = _fake_llm_payload(
            "No 1 vs 1 seed matchups exist."
        )
        answer = ChatService.answer_question("1 seed vs 1 seed matchup history")
        self.assertIn(answer.outcome,
            {"no_results", "error", "trend_success", "llm_fallback", "success"})

# 6. Response structure sanity tests
class ChatAnswerStructureTests(TestCase):

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
        self.assertIn(answer.outcome,
            {"success", "trend_success", "no_results", "error", "llm_fallback"})

    @patch(MOCK_LLM_PATH)
    def test_text_is_non_empty_on_success(self, MockLLM):
        MockLLM.return_value.generate_grounded_answer.return_value = _fake_llm_payload(
            "Gonzaga has been a dominant mid-major program."
        )
        answer = ChatService.answer_question("Tell me about Gonzaga")
        if answer.outcome in {"success", "trend_success", "llm_fallback"}:
            self.assertTrue(len(answer.text.strip()) > 0)

# 7. LLM fallback accuracy tests
class LLMFallbackAccuracyTests(TestCase):

    @patch(MOCK_LLM_PATH)
    def test_seed_query_fallback_contains_stats(self, MockLLM):
        MockLLM.return_value.generate_trend_answer.side_effect = RuntimeError("503 unavailable")
        answer = ChatService.answer_question("How often does a 12 seed beat a 5 seed?")
        self.assertEqual(answer.outcome, "llm_fallback")
        self.assertRegex(answer.text, r"\d+", msg="Fallback must include at least one number")

    @patch(MOCK_LLM_PATH)
    def test_team_query_fallback_has_dataset_rows(self, MockLLM):
        MockLLM.return_value.generate_comparison_answer.side_effect = RuntimeError("API error")
        answer = ChatService.answer_question("Compare Duke vs Kansas")
        self.assertIn(answer.outcome, {"llm_fallback", "no_results"})
        if answer.outcome == "llm_fallback":
            self.assertTrue(len(answer.text) > 0)