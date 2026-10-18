import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.recommender.config import load_settings
from src.recommender.rag import RagPipeline


class ChatContextTests(unittest.TestCase):
    def test_gradio_text_blocks_preserve_conversation(self):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "What is Assassination Classroom about?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It follows a class of students."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who are the characters?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Which anime?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Thinking..."}]},
            {"role": "user", "content": [{"type": "file", "file": {"path": "image.png"}}]},
        ]
        recent = RagPipeline.recent_history(history)
        self.assertEqual(len(recent), 4)
        self.assertIn("Assassination Classroom", recent[0]["content"])
        self.assertEqual(recent[2]["content"], "Who are the characters?")
        self.assertIsInstance(history[0]["content"], list)
        self.assertEqual(RagPipeline.recent_history(recent), recent)

    def test_history_is_bounded_and_filters_status_messages(self):
        history = [{"role": "user", "content": str(i) * 3000} for i in range(20)]
        history += [{"role": "assistant", "content": "Thinking..."},
                    {"role": "assistant", "content": "Unable to answer: offline"}]
        recent = RagPipeline.recent_history(history)
        self.assertLessEqual(len(recent), 12)
        self.assertLessEqual(sum(len(m["content"]) for m in recent), 8000)
        self.assertTrue(all(len(m["content"]) <= 2000 for m in recent))
        self.assertEqual(recent[-1]["content"], history[-3]["content"][:2000])
        self.assertEqual(len(history), 22)
        self.assertEqual(len(RagPipeline.recent_history(history[:1] * 20)), 4)

    def test_follow_up_uses_history_and_next_chat_is_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "vectors.npy", np.ones((1, 2)))
            (root / "chunks.json").write_text(json.dumps([
                {"source": "anime.md", "text": "Cowboy Bebop follows bounty hunters."}
            ]), encoding="utf-8")
            pipeline = RagPipeline.__new__(RagPipeline)
            pipeline.settings = replace(load_settings(), index=root, documents=root / "documents")
            queries, prompts = [], []
            def embed(texts):
                queries.extend(texts)
                return np.ones((1, 2))
            def answer(prompt, *args):
                prompts.append(prompt)
                return "An answer"
            pipeline.models = SimpleNamespace(embed=embed, answer=answer)
            history = [{"role": "user", "content": "What is Cowboy Bebop about?"},
                       {"role": "assistant", "content": "It follows bounty hunters."}]
            self.assertEqual(pipeline.ask("Who are they?", history), "An answer")
            self.assertIn("Cowboy Bebop", queries[-1])
            self.assertIn("It follows bounty hunters.", prompts[-1])
            self.assertIn("Question: Who are they?", prompts[-1])
            self.assertIn("[anime.md]", prompts[-1])
            # Match the Chatbot's structured callback payload, not just the
            # string messages used by CLI callers.
            blocks = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                      for m in history]
            pipeline.ask("Who are they?", blocks)
            self.assertIn("Cowboy Bebop", queries[-1])
            self.assertIn("It follows bounty hunters.", prompts[-1])
            pipeline.ask("A new question")
            self.assertEqual(queries[-1], "A new question")
            self.assertNotIn("Previous conversation", prompts[-1])
            self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
