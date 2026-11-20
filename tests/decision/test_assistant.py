"""The chat turn, driven by a fake Gemini client: no API call, no cost. Checks the loop runs each function call
through the menu, hands the result back, stops at the call limit, and never invents a function."""
from types import SimpleNamespace

from google.genai import types

from decision import assistant
from decision.assistant import MAX_CALLS, TOOLS, run_turn


def reply(text=None, calls=(), tokens=(10, 5)):
    fcs = [SimpleNamespace(name=n, args=a) for n, a in calls]
    content = types.Content(role="model", parts=[types.Part(text=text or "calling")])
    return SimpleNamespace(text=text, function_calls=fcs or None,
                           candidates=[SimpleNamespace(content=content)],
                           usage_metadata=SimpleNamespace(prompt_token_count=tokens[0], candidates_token_count=tokens[1],
                                                          thoughts_token_count=0))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.sent.append(list(contents))
        return self.replies.pop(0)


def function_responses(contents):
    return [p.function_response for c in contents for p in (c.parts or []) if p.function_response]


def test_a_question_runs_the_form_through_the_menu_and_hands_the_result_back(lab):
    form = {"how": "top", "measure": "redemption_chance", "campaign": 1, "top_n": 3}
    client = FakeClient([reply(calls=[("answer", form)]), reply(text="Mail households 1, 2 and 3.")])
    turn = run_turn(client, lab, [], "best 3 for campaign 1?")
    assert turn.answer == "Mail households 1, 2 and 3."
    assert [name for name, _, _ in turn.calls] == ["answer"]
    assert turn.households["household_key"].tolist() == [1, 2, 3]
    sent_back = function_responses(client.sent[1])
    assert sent_back[0].name == "answer" and sent_back[0].response["status"] == "answer"
    assert turn.tokens == {"input": 20, "output": 10}


def test_a_clarifying_question_from_the_code_reaches_the_model(lab):
    client = FakeClient([reply(calls=[("answer", {"how": "top", "measure": "uplift", "top_n": 5})]),
                         reply(text="Which campaign?")])
    turn = run_turn(client, lab, [], "top 5 for uplift")
    assert turn.calls[0][2].status == "ask"
    assert "Which campaign" in function_responses(client.sent[1])[0].response["message"]
    assert turn.households is None


def test_a_function_that_does_not_exist_is_refused_not_run(lab):
    client = FakeClient([reply(calls=[("run_sql", {"query": "drop table"})]), reply(text="I can't do that.")])
    turn = run_turn(client, lab, [], "run this sql")
    assert turn.calls[0][2].status == "cannot"


def test_the_loop_stops_at_the_call_limit(lab):
    looping = [reply(calls=[("describe_code", {"trait": "kids"})]) for _ in range(MAX_CALLS + 2)]
    turn = run_turn(FakeClient(looping), lab, [], "keep going")
    assert len(turn.calls) == MAX_CALLS
    assert "smaller parts" in turn.answer


def test_a_plain_answer_needs_no_function_call(lab):
    turn = run_turn(FakeClient([reply(text="The Decision Lab doesn't cover the weather.")]), lab, [], "weather?")
    assert turn.calls == [] and "weather" in turn.answer


def test_history_is_sent_with_the_right_roles(lab):
    client = FakeClient([reply(text="ok")])
    run_turn(client, lab, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "next")
    assert [c.role for c in client.sent[0]] == ["user", "model", "user"]


def test_every_tool_the_model_sees_is_one_the_code_handles_and_an_empty_call_never_gives_numbers(lab):
    for tool in TOOLS:
        result = assistant.call(lab, tool["name"], {})
        assert result.status in ("ask", "cannot") and not result.numbers, tool["name"]
