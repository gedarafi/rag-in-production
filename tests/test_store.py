from src.chat.store import ChatStore, new_id, valid_id
from tests.fakes import FakeDynamo


def store():
    return ChatStore("jobs", "conversations", client=FakeDynamo())


def test_history_is_oldest_first_and_limited():
    s = store()
    for i in range(1, 9):
        s.add_turn("u", "c", f"job{i}", f"question {i}", f"answer {i}")
    history = s.history("u", "c", limit=2)
    assert history == [{"role": "user", "content": "question 7"}, {"role": "assistant", "content": "answer 7"},
                       {"role": "user", "content": "question 8"}, {"role": "assistant", "content": "answer 8"}]


def test_a_retried_job_adds_its_turn_once():
    s = store()
    assert s.add_turn("u", "c", "job1", "q", "a") == 1
    assert s.add_turn("u", "c", "job1", "q", "a") == 1
    assert s.add_turn("u", "c", "job2", "q2", "a2") == 2


def test_conversations_are_per_user():
    s = store()
    s.add_turn("alice", "c", "job1", "q", "a")
    assert s.history("bob", "c") == []


def test_job_lifecycle():
    s = store()
    job_id = new_id()
    assert valid_id(job_id)
    s.create_job(job_id, "u", new_id(), "hi", "0" * 32)
    s.update_job(job_id, "running")
    s.update_job(job_id, "done", answer="hello")
    job = s.get_job(job_id)
    assert (job["status"], job["answer"], job["trace_id"]) == ("done", "hello", "0" * 32)
    assert job["expires_at"] > job["created_at"]
