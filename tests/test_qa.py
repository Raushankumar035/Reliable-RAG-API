import pytest

from app.core.exceptions import LLMResponseError, LLMUnavailableError
from app.generation.prompts import UNAVAILABLE_MESSAGE
from app.services.qa_service import QAService, parse_llm_json
from tests.conftest import FakeLLM, llm_json

LEAVE_Q = "How many days of paid annual leave does a full-time employee get per year?"


def test_grounded_answer_with_verified_quote(qa, fake_llm):
    fake_llm.responses = [
        llm_json(
            answer="Employees get 24 days of annual leave [1].",
            citations=[{"source": 1, "quote": "Every full-time employee receives 24 days of paid annual leave per calendar year."}],
        )
    ]
    result = qa.answer(LEAVE_Q)
    assert result.answerable and result.reason == "answered"
    assert "24" in result.answer
    src = result.sources[0]
    assert src.document == "employee_handbook.md" and src.quote_verified
    assert "24 days of paid annual leave" in src.excerpt


def test_unverifiable_quote_falls_back_to_chunk_excerpt(qa, fake_llm):
    fake_llm.responses = [llm_json(answer="24 days [1]", citations=[{"source": 1, "quote": "made-up sentence not in docs"}])]
    result = qa.answer(LEAVE_Q)
    assert result.answerable
    assert result.sources[0].quote_verified is False
    assert "annual leave" in result.sources[0].excerpt.lower()


def test_multiple_sources_are_returned_for_multi_evidence_question(qa, fake_llm):
    fake_llm.responses = [
        llm_json(answer="Within 7 days [1], then annually [2].", citations=[{"source": 1, "quote": ""}, {"source": 2, "quote": ""}])
    ]
    result = qa.answer("What security training must new employees complete and how often is it repeated?")
    assert len(result.sources) == 2
    assert len({s.chunk_id for s in result.sources}) == 2


def test_llm_says_not_answerable(qa, fake_llm):
    fake_llm.responses = [llm_json(answerable=False, answer="n/a")]
    result = qa.answer("What is the annual leave policy on sabbaticals?")
    assert not result.answerable and result.reason == "not_in_documents"
    assert result.answer == UNAVAILABLE_MESSAGE and result.sources == []


def test_irrelevant_question_is_gated_without_calling_llm(qa, fake_llm):
    result = qa.answer("What is the capital of France?")
    assert not result.answerable and result.reason == "no_relevant_context"
    assert fake_llm.calls == []  # saved an LLM call and removed any chance of hallucination


def test_empty_knowledge_base(store, settings, fake_llm):
    result = QAService(store, fake_llm, settings).answer("anything at all?")
    assert not result.answerable and result.reason == "no_documents"
    assert fake_llm.calls == []


def test_answer_without_valid_citations_is_rejected(qa, fake_llm):
    fake_llm.responses = [llm_json(answer="24 days.", citations=[])]
    assert qa.answer(LEAVE_Q).reason == "ungrounded_answer"


def test_hallucinated_citation_index_is_rejected(qa, fake_llm):
    fake_llm.responses = [llm_json(answer="24 days [99].", citations=[{"source": 99, "quote": "x"}, {"source": 0, "quote": "y"}])]
    result = qa.answer(LEAVE_Q)
    assert not result.answerable and result.reason == "ungrounded_answer"


def test_fenced_json_is_parsed(qa, fake_llm):
    fake_llm.responses = ["```json\n" + llm_json(answer="24 [1]", citations=[{"source": 1, "quote": ""}]) + "\n```"]
    assert qa.answer(LEAVE_Q).answerable


def test_invalid_json_triggers_one_retry_then_succeeds(qa, fake_llm):
    fake_llm.responses = ["Sure! The answer is 24 days.", llm_json(answer="24 [1]", citations=[{"source": 1, "quote": ""}])]
    assert qa.answer(LEAVE_Q).answerable
    assert len(fake_llm.calls) == 2


def test_persistently_invalid_json_raises(qa, fake_llm):
    fake_llm.responses = ["not json", "still not json"]
    with pytest.raises(LLMResponseError):
        qa.answer(LEAVE_Q)


def test_llm_provider_failure_raises_unavailable(qa):
    qa._llm = FakeLLM([ConnectionError("groq down")])
    with pytest.raises(LLMUnavailableError):
        qa.answer(LEAVE_Q)


def test_prompt_contains_numbered_context_and_question(qa, fake_llm):
    fake_llm.responses = [llm_json(answerable=False)]
    qa.answer(LEAVE_Q)
    system, user = fake_llm.calls[0]
    assert "ONLY" in system.content
    assert "[1] (source:" in user.content and LEAVE_Q in user.content


def test_top_k_is_capped_by_settings(qa, fake_llm, settings):
    fake_llm.responses = [llm_json(answerable=False)]
    qa.answer(LEAVE_Q, top_k=1000)
    user = fake_llm.calls[0][1].content
    assert user.count("(source:") <= settings.max_top_k


def test_parse_llm_json_variants():
    assert parse_llm_json('noise {"answerable": true, "answer": "x"} trailing')["answer"] == "x"
    with pytest.raises(ValueError):
        parse_llm_json('{"foo": 1}')
